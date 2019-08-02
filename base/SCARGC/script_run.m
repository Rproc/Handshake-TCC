acc_vec = zeros(1, 8);
time_vec = zeros(1, 8);
acc_kpca = zeros(1, 8);
time_kpca = zeros(1, 8);
index = zeros(1, 8);
j = 1;
for i = [2 4 6 8 10 15 25 50]
    [pred, vet, acc, tempo] = SCARGC_1NN('/home/procopio/Documents/Handshake-TCC/datasets/sea.data',300,150, i);
    acc_vec(j) = acc;
    time_vec(j) = acc;
    [pred2, vet2, acc2, tempo2] = SCARGC_1NN_KPca('/home/procopio/Documents/Handshake-TCC/datasets/sea.data',300,150, i);
    acc_kpca(j) = acc2;
    time_kpca = tempo2;
    index(j) = i;
    
    j = j + 1;
end
a = {'With KernelPCA', 'Without KernelPCA'};
kpca = vertcat(index, acc_kpca);
normal = vertcat(index, acc_vec);
fid = fopen('/home/procopio/Documents/Scargc_kpca_sea.txt', 'w');
fprintf(fid, '%s\n', a{1});
fprintf(fid, '%d, %.5f\n', kpca);
fprintf(fid, '%s\n', a{end});
fprintf(fid, '%d, %.5f\n', normal);

fclose(fid);