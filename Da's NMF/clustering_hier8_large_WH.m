function clustering_hier8_large_WH(dataset, X, fix_k, label, label_names, voc, prefix, suffix, random_seed_file, random_run, alg_group, alg_matrix, alg_init, varargin)

% varargin: list of data/operations for each algorithm.
% For each argument 'alg_details' in varargin:
% alg_details.matrix -> which matrix in alg_matrix to work on
% alg_details.init -> which matrix in alg_init is used for initializations
% alg_details.skip -> whether to skip the execution of function
% alg_details.func -> the function pointer to the algorithm
% alg_details.hier -> whether the function itself generates cluster hierarchies
% alg_details.init_for_hier -> the function pointer that extracts initialization for mid-level nodes
% alg_details.computeobj -> whether the final objective value needs to be computed

if ~isempty(prefix)
	prefix = [prefix, '_'];
end
if ~isempty(suffix)
	suffix = ['_', suffix];
end
if isempty(random_seed_file)
	random_seed_file = [prefix, 'random_seed', suffix, '.mat'];
end
if exist(random_seed_file, 'file')
	load(random_seed_file);
else
	s = RandStream.create('mt19937ar','seed',sum(100*clock));
	save(random_seed_file, 's');
end

if length(label) ~= 0
	ground_level = length(label);
else
	ground_level = 0;
	k = fix_k;
end
label_subset = cell(1, ground_level);
for l = 1 : ground_level
	k = length(unique(label{l}));
	label_subset{l} = cell(1, k);
	for j = 1 : k
		label_subset{l}{j} = find(label{l} == j);
	end
end

total_g = k-1;
total_run = random_run;
total_alg = length(varargin);
total_matrix = length(alg_matrix);
total_init = length(alg_init);

final_time_summary = zeros(total_run, total_alg, total_g);
final_iter_summary = zeros(total_run, total_alg, total_g);
final_error_summary = zeros(total_run, total_alg, total_g);
final_nmi_summary = cell(total_run, total_alg, total_g);
final_entropy_summary = cell(total_run, total_alg, total_g);
final_voc_summary = cell(total_run, total_alg, total_g);
final_count_summary = cell(total_run, total_alg, total_g);
final_explain_summary = cell(total_run, total_alg, total_g);
final_percent_summary = cell(total_run, total_alg, total_g);
final_priority_summary = cell(total_run, total_alg, total_g);
final_cluster_summary = cell(total_run, total_alg, total_g);
final_WH_summary = cell(total_run, total_alg, total_g);
final_cohere10_summary = zeros(total_run, total_alg, total_g);
final_cohere20_summary = zeros(total_run, total_alg, total_g);
final_unique10_summary = zeros(total_run, total_alg, total_g);
final_unique20_summary = zeros(total_run, total_alg, total_g);

if isempty(alg_group)
	alg_group = ones(1, total_alg);
end

	reset(s);
	RandStream.setGlobalStream(s);
	[m, n] = size(X);
		working_matrix = cell(1, total_matrix);
		for j = 1 : total_matrix
			working_matrix{j} = alg_matrix{j}(X);
		end
		X = X';
		group_num = length(unique(alg_group));
		for group = 1 : group_num
			for run = 1 : random_run
				index = run;
				working_init = cell(total_matrix, total_init);
				for alg = find(alg_group == group)
					alg_details = varargin{alg};
					if isempty(working_init{alg_details.matrix, alg_details.init})
						working_init{alg_details.matrix, alg_details.init} = alg_init{alg_details.init}(working_matrix{alg_details.matrix}, k);
					end
					if isfield(alg_details, 'skip') & alg_details.skip
						continue;
					end
					if alg_details.hier
						[clusters, objs, timings, Ws, priorities] = alg_details.func(working_matrix{alg_details.matrix}, k, working_init{alg_details.matrix, alg_details.init}, alg_details.computeobj);
						for i = 2 : k
							final_time_summary(index, alg, i-1) = timings(i-1);
							final_error_summary(index, alg, i-1) = objs(i-1);
							final_priority_summary{index, alg, i-1} = priorities{i-1};
							cluster = clusters{i-1};
							if size(cluster, 1) == 1
								cluster = cluster';
							end
							final_cluster_summary{index, alg, i-1} = cluster;
							cluster = cluster + 1;
							level_cluster = zeros(n, 1);
							next_label = 0;
							for j = 1 : length(unique(cluster(cluster > 0)))
								next_label = min(cluster(cluster > next_label));
								level_cluster(cluster == next_label) = j;
							end
							nmi = zeros(1, ground_level);
							for l = 1 : ground_level
								nmi(l) = compute_nmi(label{l}, level_cluster);
							end
							final_nmi_summary{index, alg, i-1} = nmi;
							entropy = zeros(1, ground_level);
							for l = 1 : ground_level
								num_labels = length(label_subset{l});
								for j = 1 : num_labels
									entropy(l) = entropy(l) + get_entropy(level_cluster(label_subset{l}{j}));
								end
								entropy(l) = entropy(l) / num_labels;
							end
							final_entropy_summary{index, alg, i-1} = entropy;
							final_voc_summary{index, alg, i-1} = cell(10, max(level_cluster));
							for j = 1 : max(level_cluster)
								[sorted, idx] = sort(Ws{i-1}(:, j), 'descend');
								max_idx = min(length(find(sorted~=0)), 10);
								final_voc_summary{index, alg, i-1}(1:max_idx, j) = voc(idx(1:max_idx));
							end
							final_count_summary{index, alg, i-1} = accumarray(level_cluster(level_cluster > 0), 1);
							final_explain_summary{index, alg, i-1} = cell(k, max(level_cluster));
							final_percent_summary{index, alg, i-1} = cell(k, max(level_cluster));
							temp_count = zeros(1, max(level_cluster));
							for l = 1 : ground_level
								num_labels = length(label_subset{l});
								for j = 1 : num_labels
									[most_freq, most_freq_count] = mode(level_cluster(label_subset{l}{j}));
									if most_freq > 0 & most_freq_count > length(label_subset{l}{j}) * 0.5 & is_label_covered(label_names{l}{j}, final_explain_summary{index, alg, i-1}(1:temp_count(most_freq), most_freq)) == false
										temp_count(most_freq) = temp_count(most_freq) + 1;
										final_explain_summary{index, alg, i-1}{temp_count(most_freq), most_freq} = label_names{l}{j};
										final_percent_summary{index, alg, i-1}{temp_count(most_freq), most_freq} = most_freq_count / length(label_subset{l}{j});
									end
								end
							end
							final_cohere10_summary(index, alg, i-1) = compute_coherent(X, Ws{i-1}, 10);
							final_cohere20_summary(index, alg, i-1) = compute_coherent(X, Ws{i-1}, 20);
							final_unique10_summary(index, alg, i-1) = find_unique_words(Ws{i-1}, 10);
							final_unique20_summary(index, alg, i-1) = find_unique_words(Ws{i-1}, 20);
							result_file = [prefix, 'data', suffix, '.mat'];
							save(result_file, '*_summary','-v7.3');
							fprintf('%s: finish run %d alg %d k %d\n', prefix(1:end-1), run, alg, i);
						end
					else
						for i = 2 : k
if i ~= k
continue
end
							temp_k = i;
							init_for_hier = alg_details.init_for_hier(working_init{alg_details.matrix, alg_details.init}, temp_k);
							[cluster, iter, obj, timing, W, H] = alg_details.func(working_matrix{alg_details.matrix}, temp_k, init_for_hier, alg_details.computeobj, dataset, index);
							final_time_summary(index, alg, temp_k-1) = timing;
							final_iter_summary(index, alg, temp_k-1) = iter;
							final_error_summary(index, alg, temp_k-1) = obj.obj;
							final_WH_summary{index, alg, temp_k-1}.W = W;
							final_WH_summary{index, alg, temp_k-1}.H = H;
							if size(cluster, 1) == 1
								cluster = cluster';
							end
							final_cluster_summary{index, alg, temp_k-1} = cluster;
							nmi = zeros(1, ground_level);
							for l = 1 : ground_level
								nmi(l) = compute_nmi(label{l}, cluster);
							end
							final_nmi_summary{index, alg, temp_k-1} = nmi;
							entropy = zeros(1, ground_level);
							for l = 1 : ground_level
								num_labels = length(label_subset{l});
								for j = 1 : num_labels
									entropy(l) = entropy(l) + get_entropy(cluster(label_subset{l}{j}));
								end
								entropy(l) = entropy(l) / num_labels;
							end
							final_entropy_summary{index, alg, temp_k-1} = entropy;
							final_voc_summary{index, alg, temp_k-1} = cell(10, temp_k);
							for j = 1 : temp_k
								if ~isnan(max(W(:, j)))
									[sorted, idx] = sort(W(:, j), 'descend');
									max_idx = min(length(find(sorted~=0)), 10);
									final_voc_summary{index, alg, temp_k-1}(1:max_idx, j) = voc(idx(1:max_idx));
								end
							end
							final_count_summary{index, alg, temp_k-1} = accumarray(cluster, 1);
							final_explain_summary{index, alg, temp_k-1} = cell(k, temp_k);
							final_percent_summary{index, alg, temp_k-1} = cell(k, temp_k);
							temp_count = zeros(1, temp_k);
							for l = 1 : ground_level
								num_labels = length(label_subset{l});
								for j = 1 : num_labels
									[most_freq, most_freq_count] = mode(cluster(label_subset{l}{j}));
									if most_freq_count > length(label_subset{l}{j}) * 0.5 & is_label_covered(label_names{l}{j}, final_explain_summary{index, alg, temp_k-1}(1:temp_count(most_freq), most_freq)) == false
										temp_count(most_freq) = temp_count(most_freq) + 1;
										final_explain_summary{index, alg, temp_k-1}{temp_count(most_freq), most_freq} = label_names{l}{j};
										final_percent_summary{index, alg, temp_k-1}{temp_count(most_freq), most_freq} = most_freq_count / length(label_subset{l}{j});
									end
								end
							end
							final_cohere10_summary(index, alg, i-1) = compute_coherent(X, W, 10);
							final_cohere20_summary(index, alg, i-1) = compute_coherent(X, W, 20);
							final_unique10_summary(index, alg, i-1) = find_unique_words(W, 10);
							final_unique20_summary(index, alg, i-1) = find_unique_words(W, 20);
							result_file = [prefix, 'data', suffix, '.mat'];
							save(result_file, '*_summary','-v7.3');
							fprintf('%s: finish run %d alg %d k %d\n', prefix(1:end-1), run, alg, temp_k);
						end
					end
				end
			end
		end

	mean_time_summary = squeeze(mean(final_time_summary))';
	mean_error_summary = squeeze(mean(final_error_summary))';
	mean_cohere10_summary = squeeze(mean(final_cohere10_summary))';
	mean_cohere20_summary = squeeze(mean(final_cohere20_summary))';
	mean_unique10_summary = squeeze(mean(final_unique10_summary))';
	mean_unique20_summary = squeeze(mean(final_unique20_summary))';

	min_error_summary = squeeze(min(final_error_summary))';
	max_cohere10_summary = squeeze(max(final_cohere10_summary))';
	max_cohere20_summary = squeeze(max(final_cohere20_summary))';
	max_unique10_summary = squeeze(max(final_unique10_summary))';
	max_unique20_summary = squeeze(max(final_unique20_summary))';

	mean_nmi_summary = cell(total_g, total_alg);
	mean_entropy_summary = cell(total_g, total_alg);
	max_nmi_summary = cell(total_g, total_alg);
	max_entropy_summary = cell(total_g, total_alg);
	
	if ground_level ~= 0
		for g = 1 : total_g
			for alg = 1 : total_alg
				mean_nmi_summary{g, alg} = mean(cell2mat(final_nmi_summary(:, alg, g)));
				mean_entropy_summary{g, alg} = mean(cell2mat(final_entropy_summary(:, alg, g)));
				max_nmi_summary{g, alg} = max(cell2mat(final_nmi_summary(:, alg, g)));
				max_entropy_summary{g, alg} = max(cell2mat(final_entropy_summary(:, alg, g)));
			end
		end
	end

	result_file = [prefix, 'data', suffix, '.mat'];
	save(result_file, '*_summary','-v7.3');

%-------------------------------------------

function covered = is_label_covered(label_name, cur_node)

i = 1;
while i <= length(cur_node)
	if strfind(cur_node{i}, label_name) == 1
		break;
	end
	i = i + 1;
end

if i > length(cur_node)
	covered = false;
else
	covered = true;
end

%-------------------------------------------

function avg_cohere = compute_coherent(X, W, top)

epsilon = 1;
k = size(W, 2);

pattern = (X ~= 0);
df = full(sum(pattern));

cohere = zeros(1, k);
for j = 1 : k
	if isnan(max(W(:, j)))
		continue;
	end
	[sorted, idx] = sort(W(:, j), 'descend');
	max_idx = min(length(find(sorted~=0)), top);
	for i = 1 : max_idx
		for h = i+1 : max_idx
			if i ~= h
				co_df = length(find(pattern(:, idx(i)) .* pattern(:, idx(h))));
				cohere(j) = cohere(j) + log((co_df + epsilon) / df(idx(i))) + log((co_df + epsilon) / df(idx(h)));
			end
		end
	end
end
avg_cohere = mean(cohere);

%-------------------------------------------

function avg_unique = find_unique_words(W, top)

k = size(W, 2);
top_words = cell(1, k);
for j = 1 : k
	if isnan(max(W(:, j)))
		continue;
	end
	[sorted, idx] = sort(W(:, j), 'descend');
	max_idx = min(length(find(sorted~=0)), top);
	top_words{j} = idx(1:max_idx);
end

num_unique = zeros(1, k);
for j = 1 : k
	pool = [];
	for i = 1 : k
		if i ~= j
			pool = union(pool, top_words{i});
		end
	end
	num_unique(j) = length(setdiff(top_words{j}, pool));
end
avg_unique = mean(num_unique);
