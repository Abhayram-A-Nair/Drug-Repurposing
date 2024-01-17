# Network-based drug repurposing using network Proximity
We propose an improved version of Similarity Network Fusion Neural Network (SNF-NN) called Similarity Network Fusion Network Proximity Neural Network (SNF-NPNN) for the task of drug repurposing. Our approach includes network-based information on drug-target and disease-target genes in the SNF-NN pipeline, resulting in significant performance enhancement compared to the original model. We also utilized less sparse labels obtained from knowledge graphs during training, improving the performance of our model. Our method shows promising results for the drug-disease link prediction task. 

# File Structure

📦 Codebase    
 ┣ 📂 Biosnap    
 ┃ ┣ 📂 code    
 ┃ ┃ ┣ 📜 calc.py    
 ┃ ┃ ┣ 📜 Convert_to_entrez.ipynb     
 ┃ ┃ ┣ 📜 DrugDiseaseInteractions_from_matrix.ipynb    
 ┃ ┃ ┣ 📜 DrugDiseaseInteractions.ipynb    
 ┃ ┃ ┣ 📜 Genelist.ipynb    
 ┃ ┃ ┣ 📜 get_reference.ipynb    
 ┃ ┃ ┣ 📜 model_input.ipynb    
 ┃ ┃ ┗ 📜 SimMat.ipynb    
 ┃ ┗ 📂 data (can be downloaded from https://snap.stanford.edu/biodata/)    
 ┣ 📂 code    
 ┃ ┣ 📜 data_convert.ipynb    
 ┃ ┣ 📜 get_disease_name.ipynb    
 ┃ ┣ 📜 SimilaritySelection_df.ipynb    
 ┃ ┣ 📜 SNF_df.ipynb    
 ┃ ┣ 📜 SNF.ipynb    
 ┃ ┗ 📜 SNFNN-final.ipynb    
 ┗ 📂 data (can be downloaded from SNF-NN https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03950-3)    
 ┗ 📜 README.md    
 
 # File contents
 📜 calc.py - script to use multiple cpus to compute the network distance metric given the labels    
 📜 Convert_to_entrez.ipynb - code to read thorugh the knowledge graph to generate drug-gene dictionary and disease-gene dictionary     
 📜 DrugDiseaseInteractions_from_matrix.ipynb - creates the labels that will be used for training from SND dataset    
 📜 DrugDiseaseInteractions.ipynb - creates the labels that will be used for training from knowlege Graphs    
 📜 Genelist.ipynb - create drug target genes and disease target genes dictionary    
 📜 get_reference.ipynb - creates a reference dataframe based on the biosnap supplementary data    
 📜 model_input.ipynb - creates a numpy object containing labels and the proximity information    
 📜 SimMat.ipynb - subsets the similarity matrices based on available drugs and diseases    
     
 📜 data_convert.ipynb - converts the downloaded dataset to usable format    
 📜 get_disease_name.ipynb - api to search disease name in UMLS meta-thesaurus and get the standard disease IDs from concept IDs     
 📜 SimilaritySelection_df.ipynb - calculates the entropy and selects the similarity metrices for SNF. It also computes pairwise similarity between all the similarity metrices    
 📜 SNF_df.ipynb - performs similarity network fusion on a folder with similarity matrix dataframes    
 📜 SNF.ipynb - performs similarity network fusion on a folder with similarity matrix numpy objects provided in SNF-NN    
 📜 SNFNN-final.ipynb - the training script that runs cross-validation to train the MLP model along with the evaluation scripts     
 
