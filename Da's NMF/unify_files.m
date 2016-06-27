X = importdata('TFIDF_Location_barcSample05.mat');
X = X';
data = importdata('voc_Location_barcSample05.mat');
voc = cellstr(data);
doc_subset = 1:370592;
term_subset = 1:31224;
save(['barc_10sample05.mat'], 'X', 'voc', 'term_subset', 'doc_subset');
