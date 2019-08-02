acc_kpca = zeros(1, 8);
time_kpca = zeros(1, 8);
index = zeros(1, 8);
j = 1;

acc = 0;
rbf = 0;
nn = 0;
fid = fopen('/home/procopio/Documents/Lux_kpca_benchmark.txt', 'a');

for i = [2 4 6 8 10 15 25 50]
    
    for k = [1 1.5 2.0 2.5 3.0 5.0 7.5 10]
        [pred2, vet2, acc2, tempo2] = SCARGC_1NN_KPca('/home/procopio/Documents/Handshake-TCC/datasets/LUweka.csv',95,150, i, k);
        
        if acc2 > acc
            acc = acc2;
            nn = i;
            rbf = k;
        end
    end
    fprintf(fid, 'knn: %d, ', nn);
    fprintf(fid, 'gamma: %.2f, ', rbf);
    fprintf(fid, 'gamma: %.5f\n\n', acc);

    acc = 0;
    rbf = 0;
    nn = 0;
end


fclose(fid);