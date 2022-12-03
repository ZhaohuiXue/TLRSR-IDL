clear
clc
close all
%% ---------------------------Settings-------------------------------------
% Load data
load data\Indian.mat;
% Add library
addpath funs\
% Random sampling
ran_sampling_rate = 0.1;
% Parameter settings
lamda1 = 1;
lamda2 = 20;
s_threshold = 0.9;
win_org = 7;
% Corresponding to K=5
dict_iter = 5;
win_change = win_org+4;
max_iter = 30;

%% -----------------------------Main Program---------------------------------------
Main_P(Indian, Indian_gt, ran_sampling_rate, lamda1,lamda2,s_threshold, win_org, dict_iter,win_change, max_iter)

%%
function [] = Main_P(Indian, Indian_gt, ran_sampling_rate, lamda1,lamda2, ...
   s_threshold, win_org, dict_iter,win_change, max_iter)
% Build the initial dictionary
[D,class_row,class_column,training_row, training_column,Indian] = build_dict(Indian, Indian_gt, ran_sampling_rate);
% Save the initial dictionary index
class_row_init = class_row;
class_column_init = class_column;          
training_row_init = training_row;          
training_column_init = training_column;   
% Iterative loop
for current_iter = 1:dict_iter+1
    disp(['Start building classification map-',num2str(current_iter-1),'...']);
    % Data rotation
    D = tensor_trans(D);
    Indian = tensor_trans(Indian);
    % Solving TLRSR(IDL) model 
    Z = TLRSR(Indian, D, max_iter,lamda1, lamda2, 21, 1);  
    %Z = cal_coefficient(Indian, D, max_iter,lamda1, lamda2, 21, 1);  
    % Denoising
    re_indian =  tprod(D, Z);
    re_indian = tensor_trans(re_indian);
    % HSI restoration
    [~, Re_by] = HSI_restoration(D,Z,Indian_gt,class_row, class_column );
    % Neighborhood regularization
    Indian = tensor_trans(Indian);
    [~, ~, neighbord_pixel,~] =  neighborhood_reg(Indian, Indian_gt, win_org,s_threshold);
    % Update window(corresponding to k=5) 
    if win_change < win_org
       win_org = win_change;
    end
    if win_change < 5
       win_change = 3;
    else
       win_change = win_change -2;
    end   
    % Classify
    predict_map =classification_fun(re_indian, Indian_gt, Re_by, neighbord_pixel);
    % Evaluate the current result
    acc_OA = accuracy_evaluation(predict_map, Indian_gt,training_row_init, training_column_init);
    predict_map = reshape(predict_map,size(Indian_gt,1),size(Indian_gt,2));
    for i = 1:size(training_column_init,2)
        predict_map(training_row_init(i), training_column_init(i) ) = Indian_gt(training_row_init(i), training_column_init(i));
    end
    disp(['ClassifictionMap-',num2str(current_iter-1),'...','     OA= ',num2str(acc_OA*100),'%']);
    disp('------------------------------------------------------------------')
    % Data rotation
    Z = tensor_trans(Z);
    D = tensor_trans(D);
    % Determine whether the maximum number of iterations has been reached
    if current_iter == dict_iter+1
        break;
    end  
    %% Update dict 
    [~, ~,neighbord_pixel,~] = neighborhood_reg(Indian, Indian_gt, win_change, s_threshold);
    class_row = class_row_init;
    class_column = class_column_init;
    for i = 1:size(Indian_gt,1)
        for j = 1:size(Indian_gt,2)
            neighbord_ij = neighbord_pixel(i,j).id;
            if neighbord_ij(1) ==0
                continue;
            end
            neighbord_ij = predict_map(neighbord_ij);
            which_class = neighbord_ij(1);
            if  (sum(neighbord_ij == which_class)) == (size(neighbord_ij,2))
               class_row(which_class).index = [class_row(which_class).index, i];
               class_column(which_class).index = [class_column(which_class).index,j];
            end 
        end
    end
    training_row = [];
    training_column = [];
    D = zeros(size(D));
    for i = 1:max(Indian_gt(:))
        training_row = [training_row, class_row(i).index];
        training_column = [training_column, class_column(i).index];
    end
    % Build a new dict
    for i =1:size(training_row,2)
        D(training_row(i), training_column(i),:) = Indian(training_row(i), training_column(i),:);
    end
end
end


