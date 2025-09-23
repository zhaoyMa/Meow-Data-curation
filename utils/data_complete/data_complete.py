import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import pickle

# Check GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ArxivVectorSearch:
    def __init__(self, jsonl_path, model_name="all-MiniLM-L6-v2", cache_dir="./cache"):
        """Initialize the vector search system with the specified model."""
        self.jsonl_path = jsonl_path
        self.model_name = model_name
        self.cache_dir = cache_dir
        # Load model to GPU if available
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.titles = []
        self.abstracts = []
        self.paper_ids = []
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_data(self):
        """Load titles and abstracts from the JSONL file."""
        print("Loading data from JSONL file...")
        with open(self.jsonl_path, 'r') as f:
            for line in tqdm(f):
                try:
                    paper = json.loads(line)
                    if 'title' in paper and 'abstract' in paper and paper['title'] and paper['abstract']:
                        # Clean the data - remove newlines and extra spaces
                        title = ' '.join(paper['title'].replace('\n', ' ').split())
                        abstract = ' '.join(paper['abstract'].replace('\n', ' ').split())
                        
                        self.titles.append(title)
                        self.abstracts.append(abstract)
                        self.paper_ids.append(paper.get('id', ''))
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line: {e}")
        
        print(f"Loaded {len(self.titles)} papers with titles and abstracts")
        
    def build_index(self, force_rebuild=False):
        """Build a FAISS index for the titles."""
        index_path = os.path.join(self.cache_dir, f"faiss_index_{self.model_name.replace('/', '_')}.idx")
        data_path = os.path.join(self.cache_dir, "arxiv_data.pkl")
        
        # Check if we can load from cache
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(data_path):
            print("Loading index and data from cache...")
            self.index = faiss.read_index(index_path)
            # If GPU is available, move index to GPU
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()  # Use a single GPU
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            with open(data_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.titles = cache_data['titles']
                self.abstracts = cache_data['abstracts']
                self.paper_ids = cache_data['paper_ids']
            print(f"Loaded index with {len(self.titles)} papers")
            return
        
        # If no cache or force rebuild, process the data
        if not self.titles:
            self.load_data()
        
        print("Generating embeddings for titles...")
        title_embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 128  # Increased batch size for GPU
        for i in tqdm(range(0, len(self.titles), batch_size)):
            batch = self.titles[i:i+batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=True, device=device)
            title_embeddings.append(embeddings.detach().cpu().numpy())
        
        # Concatenate all embeddings
        title_embeddings = np.vstack(title_embeddings)
        
        # Create and train the index
        print("Building FAISS index...")
        dimension = title_embeddings.shape[1]
        print(f"dimension{dimension}")
        # Create CPU index first
        cpu_index = faiss.IndexFlatL2(dimension)  # Using L2 distance
        
        # If GPU is available, move index to GPU
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()  # Use a single GPU
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index
            
        self.index.add(title_embeddings.astype(np.float32))
        
        # Save the CPU version of the index for future use
        print("Saving index and data to cache...")
        if torch.cuda.is_available():
            # Convert GPU index back to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
            
        with open(data_path, 'wb') as f:
            pickle.dump({
                'titles': self.titles,
                'abstracts': self.abstracts,
                'paper_ids': self.paper_ids
            }, f)
    def build_index_HNSW(self, force_rebuild=False):
        """Build a FAISS index using HNSW for the titles."""
        index_path = os.path.join(self.cache_dir, f"faiss_index_hnsw_{self.model_name.replace('/', '_')}.idx")
        data_path = os.path.join(self.cache_dir, "arxiv_data_hnsw.pkl")
        
        # Check if we can load from cache
        if not force_rebuild and os.path.exists(index_path) and os.path.exists(data_path):
            print("Loading index and data from cache...")
            # HNSW索引直接加载到CPU（HNSW不支持GPU）
            self.index = faiss.read_index(index_path)
            
            with open(data_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.titles = cache_data['titles']
                self.abstracts = cache_data['abstracts']
                self.paper_ids = cache_data['paper_ids']
            print(f"Loaded index with {len(self.titles)} papers")
            return
        
        # If no cache or force rebuild, process the data
        if not self.titles:
            self.load_data()
        
        print("Generating embeddings for titles...")
        title_embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 128
        for i in tqdm(range(0, len(self.titles), batch_size)):
            batch = self.titles[i:i+batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=True, device=device)
            title_embeddings.append(embeddings.detach().cpu().numpy())
        
        # Concatenate all embeddings
        title_embeddings = np.vstack(title_embeddings)
        
        # Create and build HNSW index
        print("Building FAISS HNSW index...")
        dimension = title_embeddings.shape[1]
        
        # 1. 使用HNSW索引代替FlatL2
        M = 48                  # 每个节点的连接数（控制内存和速度）
        efConstruction = 400    # 构建时参数（影响索引质量）
        
        # 创建HNSW索引（注意：HNSW只能在CPU上运行）
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = efConstruction
        
        # 2. 直接添加数据（HNSW不需要显式训练）
        self.index.add(title_embeddings.astype(np.float32))
        
        # 3. 设置搜索参数
        self.index.hnsw.efSearch = 128  # 搜索时候选列表大小        
        # 保存索引（HNSW索引始终在CPU）
        print("Saving index and data to cache...")
        faiss.write_index(self.index, index_path)
        
        with open(data_path, 'wb') as f:
            pickle.dump({
                'titles': self.titles,
                'abstracts': self.abstracts,
                'paper_ids': self.paper_ids
            }, f)
    def search(self, query, k=5):
        """Search for the k most similar abstracts to the query."""
        if self.index is None:
            self.build_index_HNSW()
        
        # Encode the query
        query_embedding = self.model.encode([query], convert_to_tensor=True, device=device)
        query_embedding = query_embedding.detach().cpu().numpy().astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return the results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.titles):
                results.append({
                    'id': self.paper_ids[idx],
                    'title': self.titles[idx],
                    'abstract': self.abstracts[idx],
                    'distance': float(distances[0][i])
                })
        
        return results

    def search_by_title(self, title, k=5, threshold=0.3):
        """Search for papers with titles similar to the given title.
        Returns the most similar paper if its distance is below threshold,
        otherwise returns None."""
        if self.index is None:
            self.build_index_HNSW()
        
        # Encode the query title
        title_embedding = self.model.encode([title], convert_to_tensor=True, device=device)
        title_embedding = title_embedding.detach().cpu().numpy().astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(title_embedding, k)
        
        # Check if any result is below the threshold
        for i, idx in enumerate(indices[0]):
            if idx < len(self.titles) and distances[0][i] < threshold:
                return {
                    'id': self.paper_ids[idx],
                    'title': self.titles[idx],
                    'abstract': self.abstracts[idx],
                    'distance': float(distances[0][i])
                }
        
        return None

def process_paper_outlines(jsonl_path, search_system, output_path):
    """
    Process paper outlines and add abstracts to references.

    Args:
        jsonl_path (str): Path to the JSONL file with paper outlines
        search_system (ArxivVectorSearch): Initialized search system
        output_path (str): Path to save the updated JSON file
    """
    # Load the JSONL file (line by line)
    print(f"Loading paper outlines from {jsonl_path}...")
    papers = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                paper = json.loads(line)
                papers.append(paper)
            except json.JSONDecodeError:
                continue
    #papers = papers[0:60]
    total_refs = 0
    found_abstracts = 0
    
    # Process each paper
    with open(output_path, 'w', encoding='utf-8') as out_f:
        print("Processing papers and searching for abstracts...")
        for paper in tqdm(papers):
            #print(paper)
            if 'ref_meta' in paper:
                for ref in paper['ref_meta']:
                    total_refs += 1
                    
                    # Skip if already has abstract
                    if 'abstract' in ref:
                        continue
                    
                    # Search by title if available
                    if 'title' in ref and ref['title']:
                        result = search_system.search_by_title(ref['title'])
                        
                        if result:
                            ref['abstract'] = result['abstract']
                            found_abstracts += 1
                        else:
                            ref['abstract'] = ""
                    else:
                        ref['abstract'] = ""
            out_f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    
    print(f"Processing complete. Found abstracts for {found_abstracts} out of {total_refs} references ({found_abstracts/total_refs*100:.2f}%).")
    return papers

def main(paper_outlines_path,output_path):
    # Path to the arXiv JSONL file
    jsonl_path = "D:\\outline_v2_abstract_extraction\\outline_v2_abstract_extraction\\arxiv-metadata-oai-snapshot.json"
    #paper_outlines_path = "paper_outlines_526_V5.jsonl"
    #output_path = "paper_outlines_with_abstracts_bmp.jsonl"
    
    # Print CUDA information
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU instead.")
    
    # Initialize the search system
    search_system = ArxivVectorSearch(jsonl_path)
    
    # Build the index (this will load from cache if available)
    search_system.build_index_HNSW()
    
    # Process the paper outlines
    process_paper_outlines(paper_outlines_path, search_system, output_path)


if __name__ == "__main__":
    #main("paper_outlines_527_V5_bpm.jsonl", "paper_outlines_with_abstracts_bmp.jsonl")
    main("input.jsonl", "input_abs.jsonl")