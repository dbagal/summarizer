import json
import os

def process_data(claims_json, abstracts_json, save_folder_path, num_records=500):

    with open(claims_json, "r") as fp:
        claims_dict = json.load(fp)
    
    with open(abstracts_json, "r") as fp:
        abstracts_dict = json.load(fp)

    claims = []
    abstracts = []

    for claim in claims_dict:
        pmids = claims_dict[claim]
        claims.append(claim)
        abstract = ""
        for pmid in pmids:
            abstract += abstracts_dict.get(pmid, None)
            if abstract is None:
                break
        abstracts.append(abstract)
                
    n = len(claims)
    
    for i in range(0,n,num_records):
        processed_data = {"inputs":abstracts[i:i+num_records], "outputs":claims[i:i+num_records]}
        fname = "data_"+str(i)+".json"

        with open(os.path.join(save_folder_path, fname), "w") as fp:
            json.dump(processed_data, fp, indent=4)


path_claims ="/Users/dhavalbagal/Downloads/Data/data_claim_5k_dict_all.json"
path_abst = "/Users/dhavalbagal/Downloads/Data/data_abst_200k_pure.json"
save = "/Users/dhavalbagal/Desktop/cse593_project/code_files/data"
process_data(path_claims, path_abst, save)
        