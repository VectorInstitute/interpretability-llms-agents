# import sys
# sys.path.append("/ibex/user/feij0a/phd_project/av-haystack/ImageBind")
import os
import math
import torch
import torch.nn.functional as F
import torchaudio
import argparse
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
# from ImageBind.imagebind import data
# from ImageBind.imagebind.models import imagebind_model
# from ImageBind.imagebind.models.imagebind_model import ModalityType
# import decord
# decord.bridge.set_bridge("torch")
# decord.VideoReader = decord.VideoReader

def get_first_k(dir_path, ext, k):
    return sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(ext)
    )[:k]
    

class AVRAG:
    
    def __init__(self, model_path = None, bsz = 128):
        
        self.bsz = bsz
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Instantiate model
        if model_path:
            self.model = imagebind_model.imagebind_huge(pretrained = False)
            self.model.load_state_dict(torch.load(model_path))
        else: # download pretrained model automatically
            self.model = imagebind_model.imagebind_huge(pretrained = True)
        self.model.eval()
        self.model.to(self.device)

        self.load_and_transform_func = {
            ModalityType.TEXT: data.load_and_transform_text,
            ModalityType.AUDIO: data.load_and_transform_audio_data,
            ModalityType.VISION: [data.load_and_transform_vision_data, data.load_and_transform_video_data]
        }
    
    @torch.no_grad()
    def encode(self, input_paths, data_type, cache = False) -> dict:
        """
        Args:
            input_paths (str or list): Paths to the input data.
            cache (bool): If True, loads the embeddings from a cache file.
        Returns:
            Dict: {
                filename: list,
                embeddings: torch.Tensor,
            }
        """
        if cache:
            assert input_paths.endswith(".pt")
            return torch.load(input_paths)
    
        if isinstance(input_paths, str):
            if os.path.isdir(input_paths):
                if data_type == ModalityType.VISION:
                    exts = (".mp4", ".jpg", ".png")
                elif data_type == ModalityType.AUDIO:
                    exts = (".wav", ".m4a")
                else:
                    exts = ()
    
                input_paths = sorted([
                    os.path.join(input_paths, f)
                    for f in os.listdir(input_paths)
                    if f.endswith(exts)
                ])
            else:
                input_paths = [input_paths]
    
        all_batches = []
    
        for start in range(0, len(input_paths), self.bsz):
            end = start + self.bsz
            input_batch = input_paths[start:end]
    
            if data_type == ModalityType.VISION:
                indice = 1 if input_batch[0].endswith(".mp4") else 0
                inputs = {
                    data_type: self.load_and_transform_func[data_type][indice](input_batch, self.device),
                }
            else:
                inputs = {
                    data_type: self.load_and_transform_func[data_type](input_batch, self.device),
                }
    
            # KEEP ON GPU
            embedding_batch = self.model(inputs)[data_type]
            all_batches.append(embedding_batch)
    
        embeddings = torch.cat(all_batches, dim=0)
    
        # ---- Safer filename handling ----
        if data_type != ModalityType.TEXT:
            filenames = [
                os.path.splitext(os.path.basename(path))[0]
                for path in input_paths
            ]
        else:
            # Avoid using raw query string as filename
            filenames = [f"text_{i}" for i in range(len(input_paths))]
    
        result = {
            "filename": filenames,
            "embeddings": embeddings
        }
    
        # Save cache on CPU only
        if data_type != ModalityType.TEXT:
            torch.save(
                {
                    "filename": filenames,
                    "embeddings": embeddings.cpu()
                },
                os.path.join(
                    os.path.dirname(os.path.dirname(input_paths[0])),
                    f"{'video' if data_type == ModalityType.VISION else 'audio'}_embeddings.pt"
                )
            )
    
        return result

    def _parse_srt(self, srt_path: str) -> str:
        lines = []
        with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    continue
                if "-->" in line:
                    continue
                lines.append(line)
        return " ".join(lines)

    @torch.no_grad()
    def encode_srt_dir(self, srt_dir: str, cache: bool = False) -> dict:
        """
        Build caption vocab from .srt files in a directory.
        Returns dict with keys: filename, embeddings
        filename is the basename without extension (same convention as audio/video).
        """
        if cache:
            # caller passes .pt file path in srt_dir in this mode
            assert srt_dir.endswith(".pt")
            return torch.load(srt_dir)
    
        srt_paths = sorted(
            os.path.join(srt_dir, f)
            for f in os.listdir(srt_dir)
            if f.endswith(".srt")
        )
        if len(srt_paths) == 0:
            raise ValueError(f"No .srt files found in {srt_dir}")
    
        texts = [self._parse_srt(p) for p in srt_paths]
    
        # Important: encode expects list[str] for TEXT
        cap_embed = self.encode(texts, ModalityType.TEXT, cache=False)
    
        # Replace filenames (currently texts) with the .srt basenames for alignment
        cap_embed["filename"] = [os.path.splitext(os.path.basename(p))[0] for p in srt_paths]
    
        # Optional cache next to srt_dir
        torch.save(cap_embed, os.path.join(os.path.dirname(srt_dir), "caption_embeddings.pt"))
        return cap_embed
        

    def topk(self, queries, vocabs, k=1, log=True):
        """
        Args:
            queries (torch.Tensor, (n, d)): Query embeddings.
            vocabs (torch.Tensor, (m, d)): Vocabulary embeddings.
            k (int): Number of top results to return.
        Returns:
            List (n, k): Top k results.
        """
        q = torch.nn.functional.normalize(queries, dim=-1)
        v = torch.nn.functional.normalize(vocabs, dim=-1)
    
        scores = q @ v.T
        values, indices = torch.topk(scores, k=k, dim=-1)
    
        if log:
            for q_idx in range(values.shape[0]):
                print(f"\nQuery {q_idx}:")
                for rank in range(k):
                    print(
                        f"  Rank {rank+1} | "
                        f"Index: {indices[q_idx, rank].item()} | "
                        f"Score: {values[q_idx, rank].item():.6f}"
                    )
    
        return indices, values


    def pair_rag(self, query=None, vocab=None, k=1):

        topk_indices, topk_values = self.topk(
            query["embeddings"],
            vocab["embeddings"],
            k=k
        )
    
        topk_files = []
    
        for q_idx, topi_indices in enumerate(topk_indices):
    
            results = []
    
            for rank, vocab_idx in enumerate(topi_indices):
    
                results.append({
                    "rank": rank + 1,
                    "file": vocab["filename"][vocab_idx],
                    "score": float(topk_values[q_idx, rank])
                })
    
            topk_files.append({
                query["filename"][q_idx]: results
            })
    
        return topk_files
        
    def joint_rag(
        self,
        query,
        vocab_vision,
        vocab_audio,
        vocab_caption,
        k=1,
        mode='0'
    ):
        """
        mode '0': paper-faithful AV-RAG with caption averaging
        """
    
        if mode != '0':
            raise NotImplementedError("Only mode '0' supported.")
    
        # ----- Align by filename -----
        mv = {f: e for f, e in zip(vocab_vision["filename"], vocab_vision["embeddings"])}
        ma = {f: e for f, e in zip(vocab_audio["filename"], vocab_audio["embeddings"])}
        mc = {f: e for f, e in zip(vocab_caption["filename"], vocab_caption["embeddings"])}
    
        common = sorted(set(mv) & set(ma) & set(mc))
        if len(common) == 0:
            raise ValueError("No overlapping filenames between vocabs.")
    
        # Move to GPU for scoring
        V = torch.stack([mv[f] for f in common], dim=0).to(self.device)
        A = torch.stack([ma[f] for f in common], dim=0).to(self.device)
        C = torch.stack([mc[f] for f in common], dim=0).to(self.device)
        q = query["embeddings"].to(self.device)
    
        # ----- Hadamard AV fusion -----
        E_av = torch.nn.functional.normalize(V * A, dim=-1)
        C = torch.nn.functional.normalize(C, dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1)
    
        # ----- Similarities -----
        S_av = q @ E_av.T
        S_cap = q @ C.T
        S_final = (S_av + S_cap) / 2
    
        values, indices = torch.topk(S_final, k=k, dim=-1)
    
        # Move back to CPU for output formatting
        values = values.cpu()
        indices = indices.cpu()
    
        topk_files = []
    
        for q_idx in range(indices.shape[0]):
            results = []
            for rank in range(indices.shape[1]):
                idx = indices[q_idx, rank].item()
                results.append({
                    "rank": rank + 1,
                    "file": common[idx],
                    "score": float(values[q_idx, rank]),
                })
            topk_files.append({query["filename"][q_idx]: results})
    
        return topk_files

    # For ablation studies
    def compute_scores_av_only(self, q, V, A):
        E_av = torch.nn.functional.normalize(V * A, dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1)
        return q @ E_av.T
    
    
    def compute_scores_caption_only(self, q, C):
        q = torch.nn.functional.normalize(q, dim=-1)
        C = torch.nn.functional.normalize(C, dim=-1)
        return q @ C.T
    
    
    def compute_scores_joint(self, q, V, A, C):
        S_av = self.compute_scores_av_only(q, V, A)
        S_cap = self.compute_scores_caption_only(q, C)
        return (S_av + S_cap) / 2

    
    #Salient frame selector (SFS)
    # -----------------------------
    # 1) Paper SFS DP (Algorithm 1)
    # -----------------------------
    def sfs_select_indices(self, Q: torch.Tensor, k: int) -> list[int]:
        """
        Paper Algorithm 1 (DP) to select k indices from m candidates given Q (m x m).
        Returns selected indices (length k) in increasing order.
        """
        assert Q.dim() == 2 and Q.shape[0] == Q.shape[1], "Q must be square (m x m)"
        m = Q.shape[0]
        assert 1 <= k <= m, f"k must be in [1, m], got k={k}, m={m}"

        # C[i][j] = min cost to end at i selecting j frames
        C = torch.full((m + 1, k + 1), float("inf"), device=Q.device)
        back = torch.full((m + 1, k + 1), -1, dtype=torch.long, device=Q.device)

        C[0, 0] = 0.0

        # i in [1..m], j in [1..k]
        for j in range(1, k + 1):
            for i in range(j, m + 1):
                # transition from p to i: p in [j-1 .. i-1]
                best_cost = float("inf")
                best_p = -1
                for p in range(j - 1, i):
                    prev = C[p, j - 1]
                    if torch.isinf(prev):
                        continue
                    # Q is 0-indexed for candidates: candidate idx = i-1, p-1
                    # For p==0 (no previous), we define 0 transition cost.
                    trans = 0.0 if p == 0 else Q[p - 1, i - 1].item()
                    cost = prev.item() + trans
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p

                C[i, j] = best_cost
                back[i, j] = best_p

        # backtrack from i=m (paper uses i<-m); you can also choose argmin_i C[i,k]
        i = m
        j = k
        result = []
        while j > 0:
            result.append(i - 1)  # candidate index
            i = back[i, j].item()
            j -= 1

        result.reverse()
        return result

    # ---------------------------------------
    # 2) Build Q = cosine_sim + temporal_penalty
    # ---------------------------------------
    def build_sfs_Q(self, z: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
        """
        z: (m, d) candidate embeddings (paper: Hadamard fused AV per sampled frame)
        Returns Q: (m, m)
        """
        assert z.dim() == 2, "z must be (m, d)"
        m = z.shape[0]

        z = F.normalize(z, dim=-1)
        Gamma = z @ z.T  # cosine similarity matrix (m x m)

        # temporal penalty Δ_ab = γ * ( 1 / sin(pi/2 * |a-b|) + 1 - 1 )
        # which simplifies to γ * (1 / sin(pi/2 * |a-b|)) for |a-b|>0, and 0 on diag.
        idx = torch.arange(m, device=z.device)
        dist = (idx[:, None] - idx[None, :]).abs().float()

        Delta = torch.zeros((m, m), device=z.device)
        nonzero = dist > 0
        # avoid division by zero by masking dist>0
        denom = torch.sin((math.pi / 2.0) * dist[nonzero])
        Delta[nonzero] = gamma * (1.0 / denom)

        Q = Gamma + Delta
        return Q

    # ---------------------------------------
    # 3) Wrapper: given candidate embeddings -> selected indices
    # ---------------------------------------
    def sfs(self, candidate_z: torch.Tensor, k: int, gamma: float = 10.0) -> list[int]:
        """
        candidate_z: (m, d) embeddings for m sampled frames
        returns: indices into the m candidates
        """
        Q = self.build_sfs_Q(candidate_z, gamma=gamma)
        return self.sfs_select_indices(Q, k=k)

if __name__ == "__main__":

    base_dir = "/fs01/projects/aixpert/users/aravind/interpretability_agent_bootcamp/implementations/multimedia_rag/data/Customer_Service_Interactions"

    video_dir = os.path.join(base_dir, "process-video")
    audio_dir = os.path.join(base_dir, "process-audio")
    caption_dir = os.path.join(base_dir, "caption")

    rag = AVRAG(model_path="./checkpoints/imagebind_huge.pth", bsz=16)

    # ---- Example Queries ----
    text_list = [
        "What inconsistency first reveals the receptionist's misleading professional approach to the customer regarding the hotel's services?",
        "Despite the customer's claims of being full after finishing a second serving of noodles, the chef insists on preparing a final rice dish, emphasizing it's part of 'Japanese culture.'",
        "McDouble"
    ]

    # Helper
    def get_first_k(dir_path, ext, k):
        return sorted(
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(ext)
        )[:k]


    video_paths = get_first_k(video_dir, ".mp4", 4)
    audio_paths = get_first_k(audio_dir, ".wav", 4)
    caption_paths = get_first_k(caption_dir, ".srt", 4)
    
    print("Using videos:", video_paths)
    print("Using audios:", audio_paths)
    print("Using captions:", caption_paths)
    
    print("\nEncoding query...")
    t_embed = rag.encode(text_list, ModalityType.TEXT)
    
    print("Encoding videos...")
    v_embed = rag.encode(video_paths, ModalityType.VISION)
    
    print("Encoding audios...")
    a_embed = rag.encode(audio_paths, ModalityType.AUDIO)
    
    print("Encoding captions...")
    texts = [rag._parse_srt(p) for p in caption_paths]
    c_embed = rag.encode(texts, ModalityType.TEXT)
    c_embed["filename"] = [
        os.path.splitext(os.path.basename(p))[0]
        for p in caption_paths
    ]

    print("\nRunning Joint AV-RAG retrieval...")
    j_res = rag.joint_rag(
        t_embed,
        v_embed,
        a_embed,
        c_embed,
        k=3
    )

    print("\n================ Joint AV-RAG Results ================")
    for result in j_res:
        print(result)

    # To begin testing SFS


    print("\n================ Running SFS on Top-1 Videos ================")

    import decord
    decord.bridge.set_bridge("torch")
    
    def sample_frames(video_path, m=16):
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = torch.linspace(0, total - 1, steps=m).long()
        frames = vr.get_batch(indices)  # (m, H, W, C)
        return frames, indices
    
    
    def encode_frames_with_imagebind(rag, frames):
        """
        Convert frame tensors -> temp jpg files -> encode using existing encode()
        """
        import tempfile
        from PIL import Image
    
        tmp_dir = tempfile.mkdtemp()
        paths = []
    
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame.numpy())
            path = os.path.join(tmp_dir, f"{i}.jpg")
            img.save(path)
            paths.append(path)
    
        v_embed = rag.encode(paths, ModalityType.VISION)
        return v_embed["embeddings"]
    
    def sample_audio_windows(audio_path, frame_indices, video_fps, window_sec=1.0):
        waveform, sr = torchaudio.load(audio_path)
        windows = []
    
        for frame_idx in frame_indices:
            center_time = frame_idx.item() / video_fps
            start = int(max(0, (center_time - window_sec/2) * sr))
            end = int(min(waveform.shape[1], (center_time + window_sec/2) * sr))
            clip = waveform[:, start:end]
            windows.append(clip)
    
        return windows
    
    # Iterate over queries
    for q_idx, result in enumerate(j_res):
    
        query_key = list(result.keys())[0]
        top1_video_id = result[query_key][0]["file"]  # e.g. '001'
    
        # find full path
        top1_path = [p for p in video_paths if top1_video_id in p][0]
    
        print(f"\nQuery: {query_key}")
        print(f"Top-1 video: {top1_video_id}")
        print(f"Video path: {top1_path}")
    
        # # ---- Sample candidate frames ----
        # frames, frame_indices = sample_frames(top1_path, m=16)
    
        # # ---- Encode frame embeddings ----
        # z = encode_frames_with_imagebind(rag, frames)
    
        # # ---- Run SFS ----
        # selected = rag.sfs(z, k=5, gamma=10.0)

        # ---- Sample frames ----
        frames, frame_indices = sample_frames(top1_path, m=16)
        
        # ---- Encode vision ----
        vision_embed = encode_frames_with_imagebind(rag, frames)
        
        # ---- Get matching audio path ----
        audio_path = [p for p in audio_paths if top1_video_id in p][0]
        
        # ---- Get fps ----
        vr = decord.VideoReader(top1_path)
        video_fps = vr.get_avg_fps()
        
        # ---- Sample audio windows ----
        audio_clips = sample_audio_windows(audio_path, frame_indices, video_fps)
        
        # Save temporary wav clips for encoding
        import tempfile
        tmp_audio_dir = tempfile.mkdtemp()
        audio_paths_tmp = []
        
        for i, clip in enumerate(audio_clips):
            path = os.path.join(tmp_audio_dir, f"{i}.wav")
            torchaudio.save(path, clip, 16000)
            audio_paths_tmp.append(path)
        
        audio_embed = rag.encode(audio_paths_tmp, ModalityType.AUDIO)["embeddings"]
        
        # ---- Fuse AV per frame ----
        z = vision_embed * audio_embed
        
        # ---- Run SFS ----
        selected = rag.sfs(z, k=5, gamma=0.0)
    
        print("Candidate frame indices:", frame_indices.tolist())
        print("Selected SFS indices (within candidates):", selected)
        print("Selected actual frame numbers:", frame_indices[selected].tolist())


        import matplotlib.pyplot as plt

        # Build Q
        Q = rag.build_sfs_Q(z, gamma=0.0)
        
        # Move to CPU for plotting
        Q_cpu = Q.detach().cpu().numpy()
        
        plt.figure(figsize=(6,5))
        plt.imshow(Q_cpu, cmap="viridis")
        plt.colorbar()
        plt.title(f"Q Matrix Heatmap - Video {top1_video_id}")
        plt.xlabel("Frame index")
        plt.ylabel("Frame index")
        plt.tight_layout()
        # plt.show()
        # Save to current directory
        save_path = f"tmp_{top1_video_id}_{query_name}.jpg"
        plt.savefig(save_path, dpi=200)
        plt.close()
        
        print(f"Saved Q heatmap to {save_path}")
    
    
    

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="AV-RAG")
#     parser.add_argument("--model_path", type=str, default="./checkpoints/imagebind_huge.pth", help="Path to the model.")
#     parser.add_argument("--bsz", type=int, default=128, help="Batch size.")
#     parser.add_argument("--cache", action="store_true", help="Use cache.")
#     parser.add_argument("--mode", type=str, default="0", help="Mode.")
#     parser.add_argument("--topk", type=int, default=1, help="Number of top results to return.")
#     parser.add_argument("--alpha_v", type=float, default=0.5, help="The importance of vision compared to audio.")
#     args = parser.parse_args()

#     rag = AVRAG(model_path = args.model_path, bsz = args.bsz)

#     text_list=["A dog", "A car", "A bird"]

#     image_paths=["./assets/test/images/dog.jpg", "./assets/test/images/car.jpg", "./assets/test/images/bird.jpg"]
#     video_paths=["./assets/test/videos/dog.mp4", "./assets/test/videos/car.mp4", "./assets/test/videos/bird.mp4"]
#     audio_paths=["./assets/test/audios/dog.wav", "./assets/test/audios/car.wav", "./assets/test/audios/bird.wav"]
#     if args.cache:
#         image_paths = "./assets/image_embeddings.pt"
#         video_paths = "./assets/video_embeddings.pt"
#         audio_paths = "./assets/audio_embeddings.pt"
    
    
#     t_embed = rag.encode(text_list, ModalityType.TEXT)
#     v_embed = rag.encode(video_paths, ModalityType.VISION, cache = args.cache)
#     a_embed = rag.encode(audio_paths, ModalityType.AUDIO, cache = args.cache)
    
#     v_res = rag.pair_rag(t_embed, v_embed, k = args.topk)
#     a_res = rag.pair_rag(t_embed, a_embed, k = args.topk)
#     j_res = rag.joint_rag(t_embed, v_embed, a_embed, k = args.topk, alpha_v = args.alpha_v, mode = args.mode)

#     print("=========================Text-Vision RAG============================")
#     print(v_res)
#     print("\n=========================Text-Audio RAG============================")
#     print(a_res)
#     print("\n==================Text-(Audio, Vision) joint RAG=====================")
#     print(j_res)
    

