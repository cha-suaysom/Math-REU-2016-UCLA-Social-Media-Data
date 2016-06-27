function mtx2mat(data_file)

X = mmread([data_file '.mtx']);
X=X';
[X, term_subset, doc_subset] = tfidf(X, 5, 3);

voc = text2cell('bagOfWords.txt', '\t');
voc = voc(term_subset);

save([data_file, '.mat'], 'X', 'voc', 'term_subset', 'doc_subset');
