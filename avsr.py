from pathlib import Path
import os
import pickle
import torch
import random
from tqdm import tqdm
from profanityfilter import ProfanityFilter
import numpy as np
from config import MyParser
import whisper
import clip
import csv
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from pathlib import Path
import numpy as np
import os

import torch
import whisper
import torchaudio.transforms as at

import csv
import editdistance
import av
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

#######
class calc_metrics:
    def __init__(self):
        pass
    def __call__(self, refs, preds):
        """
        refs are output from dataloader, so uses the collate fn, that already contains the normalization
        preds are the output of whisper tokenizer, which doesn't have dataset specific normalization

        they should both in list (list of list)
        """
        distance = 0
        tokens = 0
        wer_list = []
        processed_preds = []
        processed_refs = []
        exclude = [",", "?", ".", "!", ";"]
        for ref, pred in zip(refs, preds):
            pred = pred.lower()
            pred = ''.join(ch for ch in pred if ch not in exclude)
            processed_preds.append(pred)
            processed_refs.append(ref) # do not process ref
            cur_dist =editdistance.distance(pred.split(" "), ref.split(" "))
            cur_tokens = len(ref.split(" "))
            wer_list.append(cur_dist/cur_tokens)
            distance += cur_dist
            tokens += cur_tokens

        return {"wer":distance/tokens}, (wer_list, processed_preds, processed_refs)




def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    with av.open(wave_path, metadata_errors="ignore") as container:
        decode = container.decode(audio=0)
        first_frame = next(decode)
        cur_sample_rate = first_frame.sample_rate
        aframes_list = [first_frame.to_ndarray()]
        for frame in decode:
            aframes_list.append(frame.to_ndarray())
        aframes = np.concatenate(aframes_list, 1)
        wav = torch.as_tensor(aframes).mean(dim=0)
        if cur_sample_rate != sample_rate:
            wav = at.Resample(cur_sample_rate, sample_rate, dtype=wav.dtype)(wav)
        if wav.mean() == 0:
            print(wave_path, "empty!")
    return wav

def load_img(fn, num_img):
    if fn.endswith(".mkv"):
        img_fn = fn.replace(".mkv", f"-{num_img}.pt")
    elif fn.endswith(".mp4"):
        img_fn = fn.replace(".mp4", f"-{num_img}.pt")
    else:
        raise RuntimeError(f"video_fn extension not supported: {fn}")
    if os.path.isfile(img_fn):
        ret_frames = torch.load(img_fn, map_location="cpu")
    else:
        with av.open(fn, metadata_errors="ignore") as container:
            all_frames = [frame.to_image() for frame in container.decode(video=0)]
            mul = len(all_frames) // num_img
            ret_frames = [torch.from_numpy(np.array(f.convert("RGB"), dtype=np.float32)) for f in all_frames[::mul][:num_img]]
            ret_frames = torch.stack(ret_frames, dim=0)
            ret_frames = ret_frames.permute(0, 3, 1, 2) / 255.0
        torch.save(ret_frames, img_fn)
    return ret_frames

class VisSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, sample_rate):
        super().__init__()
        self.split = split
        self.args = args
        self.sample_rate = sample_rate
        self.data = []
        with open(Path(args.dataset_dir)/"VisSpeech.csv", "r") as file:
            csv_file = csv.reader(file)
            header = next(csv_file)
            missing = []
            for i, item in enumerate(csv_file):
                key,yt_id,start_time,end_time,text = item
                fn = Path(args.dataset_dir)/f"{key}.mkv"
                if fn.is_file():
                    self.data.append([fn, text])
                else:
                    break#音声ファイルが読み込めなくなると勝手に終了
                    fn = Path(str(fn).replace(".mkv", ".mp4"))
                    assert fn.is_file(), f"{fn} doesn't exist!"
                    self.data.append([fn, text])

            print(f"expacting {i+1} files, and get {len(self.data)} files")
            print(f"missing: {missing}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        audio_path, raw_text = self.data[id]

        # audio
        audio = load_wave(str(audio_path), sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        if self.args.socratic == "1":
            imgs = load_img(str(audio_path), num_img=self.args.num_img)
        else:
            imgs = None
        return {
            "audio_path": audio_path,
            "input_mel": mel,
            "imgs": imgs,
            "raw_text": raw_text
        }

    def collate(self, batch):
        audio_paths, input_mels, imgs, raw_text = [], [], [], []
        for f in batch:
            audio_paths.append(f['audio_path'])
            input_mels.append(f["input_mel"])
            imgs.append(f['imgs'])
            raw_text.append(f['raw_text'])


        input_mels = torch.stack(input_mels, dim=0)
        
        collated_batch = {}
        collated_batch["input_mels"] = input_mels
        collated_batch["audio_paths"] = audio_paths
        collated_batch["imgs"] =  imgs
        collated_batch["raw_text"] =  raw_text

        return collated_batch        


def get_dataloader(args):
    dataset = VisSpeechDataset(args, "test", args.sample_rate) # there is only one split, only test
    print("dataset size: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=args.batch_size, drop_last=False, shuffle=False, 
                        num_workers=args.num_workers,
                        collate_fn=dataset.collate, persistent_workers=True
                        )

    return loader
#####################
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = MyParser().parse_args()
    print(args)
    args.dataset="visspeech"
    args.model="medium.en"
    args.dataset_dir="visspeech"
    args.core_metric="wer"
    args.pk="0"
    args.ok="50"
    args.num_img=3
    args.place_txt_fn = "visspeech/categories_places365.txt"
    vsocratic="1" # "1" mean also input visual prompt utilizing CLIP, "0" mean audio only
    
    print(args)
    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################

    # clip_version = "ViT-L/14" #@param ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"] {type:"string"}
    clip_version = "ViT-L/14@336px" 

    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768, "ViT-L/14@336px": 768}[clip_version]
    clip_img_res = {'ViT-L/14': 224, "ViT-L/14@336px": 336}[clip_version]

    if args.socratic == "1":
        clip_model, _ = clip.load(clip_version)  # clip.available_models()
        preprocess = Compose([
        Resize(clip_img_res, interpolation=BICUBIC),
        CenterCrop(clip_img_res),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
        clip_model.to('cpu').eval()

        def num_params(model):
            return np.sum([int(np.prod(p.shape)) for p in model.parameters()])
        print("clip_Model parameters (total):", num_params(clip_model))
        print("clip_Model parameters (image encoder):", num_params(clip_model.visual))
        print("clip_Model parameters (text encoder):", num_params(clip_model.token_embedding) + num_params(clip_model.transformer))
        print("Input image resolution:", clip_model.visual.input_resolution)
        print("Context length:", clip_model.context_length)
        print("Vocab size:", clip_model.vocab_size)
        img_size = clip_model.visual.input_resolution

    def get_text_feats(in_text, batch_size=64):
        text_tokens = clip.tokenize(in_text).to('cpu')
        text_id = 0
        text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)
            text_batch = text_tokens[text_id:text_id+batch_size]
            with torch.no_grad():
                batch_feats = clip_model.encode_text(text_batch).float()
                batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
                batch_feats = np.float32(batch_feats.cpu())
                text_feats[text_id:text_id+batch_size, :] = batch_feats
                text_id += batch_size
        return text_feats

    def get_img_feats(img):
        assert len(img.shape) == 4
        img_in = preprocess(img)
        with torch.no_grad():
            img_feats = clip_model.encode_image(img_in.to('cpu')).float()
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        img_feats = np.float32(img_feats.cpu())
        return img_feats

    def get_nn_text(raw_texts, text_feats, img_feats, topk):
        assert len(img_feats.shape) == 2 and img_feats.shape[0] == args.num_img, f"img_feats shape: {img_feats.shape}"
        scores = []
        texts = []
        for img_feat in img_feats:
            cur_scores = text_feats @ img_feat[None,...].T
            cur_scores = cur_scores.squeeze()
            scores.append(cur_scores)
            texts += raw_texts
        scores = np.concatenate(scores) 
        high_to_low_ids = np.argsort(scores).squeeze()[::-1]
        selected_texts = []
        selected_scores = []
        for id in high_to_low_ids:
            if texts[id] in selected_texts:
                continue
            if len(selected_texts) >= topk:
                break
            selected_texts.append(texts[id])
            selected_scores.append(scores[id])
        return selected_texts, selected_scores
        

    if args.socratic == "1":
        place_fn = args.place_pkl_fn
        object_fn = args.object_pkl_fn
        if os.path.isfile(place_fn):
            print("load place texts and feats from ", place_fn)
            with open(place_fn, "rb") as f:
                place_f = pickle.load(f)
            place_texts = place_f['place_texts']
            place_feats = place_f['place_feats']
            print("length of place texts: ", len(place_texts))
        else:
            print("embed places365 text")
            # Load scene categories from Places365.
            place_categories = np.loadtxt(args.place_txt_fn, dtype=str)
            place_texts = []
            for place in place_categories:
                try:
                    place = place.split('/')[2:]
                    if len(place) > 1:
                        place = place[1] + ' ' + place[0]
                    else:
                        place = place[0]
                    place = place.replace('_', ' ')
                    place_texts.append(place)
                except:
                    pass
            place_feats = get_text_feats([f'Photo of a {p}.' for p in place_texts])
            print("length of place texts: ", len(place_texts))
            with open(place_fn, "wb") as f:
                pickle.dump({"place_texts": place_texts, "place_feats": place_feats}, f)

        # Load object categories from Tencent ML Images.
        if os.path.isfile(object_fn):
            print("load tencent ml image texts and feats from ", object_fn)
            with open(object_fn, "rb") as f:
                object_f = pickle.load(f)
            object_texts = object_f['object_texts']
            object_feats = object_f['object_feats']
            print("num of object texts: ", len(object_texts))
        else: 
            print("embed tencent ml image text")
            with open(args.object_txt_fn) as fid:
                object_categories = fid.readlines()
            object_texts = []
            pf = ProfanityFilter()
            for object_text in object_categories[1:]:
                object_text = object_text.strip()
                object_text = object_text.split('\t')[3]
                safe_list = ''
                for variant in object_text.split(','):
                    text = variant.strip()
                    if pf.is_clean(text):
                        safe_list += f'{text}, '
                safe_list = safe_list[:-2]
                if len(safe_list) > 0:
                    object_texts.append(safe_list)
                
            object_texts = [o for o in list(set(object_texts)) if o not in place_texts]  # Remove redundant categories.
            object_feats = get_text_feats([f'Photo of a {o}.' for o in object_texts])
            print("length of object texts: ", len(object_texts))
            with open(object_fn, "wb") as f:
                pickle.dump({"object_texts": object_texts, "object_feats": object_feats}, f)
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################
    ###################### CLIP textual feature embedding ######################


    ###################################

    loader = get_dataloader(args)

    model = whisper.load_model(args.model)
    model.eval()
    model.to('cpu')

    refs = []
    preds = []
    all_prompts = []
    for i, b in enumerate(tqdm(loader)):
        input_mels = b["input_mels"].half().to('cpu')
        raw_texts = b["raw_text"]
        imgs = b['imgs']
        with torch.no_grad(): 
            for input_mel, raw_text, img in zip(input_mels, raw_texts, imgs):
                if args.socratic == "1":
                    img = img.to('cpu')
                    img_feats = get_img_feats(img)
                    place_list = ''
                    if args.place_topk > 0:
                        sorted_places, places_scores = get_nn_text(place_texts, place_feats, img_feats, args.place_topk)
                        sorted_places = sorted_places[::-1]

                        for i in range(len(sorted_places)):
                            place_list += f'{sorted_places[i]}, '
                    object_list = ''
                    if args.obj_topk > 0:
                        sorted_obj_texts, obj_scores = get_nn_text(object_texts, object_feats, img_feats, args.obj_topk)
                        sorted_obj_texts = sorted_obj_texts[::-1]
                        
                        for i in range(len(sorted_obj_texts)):
                            object_list += f'{sorted_obj_texts[i].split(",")[0]}, '
                        object_list = object_list[:-2] + ". "
                    prompt = place_list + object_list
                    if len(prompt) == 0:
                        prompt = None
                else:
                    prompt = None
                all_prompts.append(prompt)

                options = whisper.DecodingOptions(task=args.task, language=args.language, without_timestamps=True, beam_size=args.beam_size, block_ngrams=args.block_ngrams, prompt=prompt)
                results = whisper.decode(model, input_mel, options)
                print(f"results.text:{results.text}")
                preds.append(results.text)
                refs.append(raw_text)


    
    inference_metrics, (wer_list, processed_preds, processed_refs) = calc_metrics()(refs, preds)
    print("results:", inference_metrics)
    print("results:", inference_metrics)
    if args.topk > 0:
        import numpy as np
        inds = np.argsort(wer_list)[::-1]
        for ind in inds[:args.topk]:
            print("-"*10)
            print("wer/mer: ", wer_list[ind])
            print("ref: ", processed_refs[ind])
            print("pred: ", processed_preds[ind])
            print("prompt: ", all_prompts[ind])
    else:
        for j, (k, v) in enumerate(zip(processed_refs, processed_preds)):
            if j % 100 == 0:
                print("-"*10)
                print("ref: ", k)
                print("pred: ", v)
    
    print("results:", inference_metrics)
    print("results:", inference_metrics)


