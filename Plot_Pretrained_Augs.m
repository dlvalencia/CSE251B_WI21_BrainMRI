vgg16_train_loss = table2array(readtable('Exp_Results/VGG16_Contrast1p5_Sharpness0p7/training_losses.txt', 'ReadVariableNames', false));
vgg16_train_loss{1,1} = vgg16_train_loss{1,1}(2:end);
vgg16_train_loss{1,end} = vgg16_train_loss{1,end}(1:end-1);
vgg16_train_loss = cell2mat(cellfun(@str2double, vgg16_train_loss, 'UniformOutput', false));

vgg16_val_loss = table2array(readtable('Exp_Results/VGG16_Contrast1p5_Sharpness0p7/val_losses.txt', 'ReadVariableNames', false));
vgg16_val_loss{1,1} = vgg16_val_loss{1,1}(2:end);
vgg16_val_loss{1,end} = vgg16_val_loss{1,end}(1:end-1);
vgg16_val_loss = cell2mat(cellfun(@str2double, vgg16_val_loss, 'UniformOutput', false));

resnet50_train_loss = table2array(readtable('Exp_Results/ResNet50_LR0.25e-3_Contrast1.5_Sharpness0.7/training_losses.txt', 'ReadVariableNames', false));
resnet50_train_loss{1,1} = resnet50_train_loss{1,1}(2:end);
resnet50_train_loss{1,end} = resnet50_train_loss{1,end}(1:end-1);
resnet50_train_loss = cell2mat(cellfun(@str2double, resnet50_train_loss, 'UniformOutput', false));

resnet50_val_loss = table2array(readtable('Exp_Results/ResNet50_LR0.25e-3_Contrast1.5_Sharpness0.7/val_losses.txt', 'ReadVariableNames', false));
resnet50_val_loss{1,1} = resnet50_val_loss{1,1}(2:end);
resnet50_val_loss{1,end} = resnet50_val_loss{1,end}(1:end-1);
resnet50_val_loss = cell2mat(cellfun(@str2double, resnet50_val_loss, 'UniformOutput', false));

densenet121_train_loss = table2array(readtable('Exp_Results/DN121_LR025_HighContrast/training_losses.txt', 'ReadVariableNames', false));
densenet121_train_loss{1,1} = densenet121_train_loss{1,1}(2:end);
densenet121_train_loss{1,end} = densenet121_train_loss{1,end}(1:end-1);
densenet121_train_loss = cell2mat(cellfun(@str2double, densenet121_train_loss, 'UniformOutput', false));

densenet121_val_loss = table2array(readtable('Exp_Results/DN121_LR025_HighContrast/val_losses.txt', 'ReadVariableNames', false));
densenet121_val_loss{1,1} = densenet121_val_loss{1,1}(2:end);
densenet121_val_loss{1,end} = densenet121_val_loss{1,end}(1:end-1);
densenet121_val_loss = cell2mat(cellfun(@str2double, densenet121_val_loss, 'UniformOutput', false));

figure()
subplot(1,3,1);
plot(vgg16_train_loss, '-r', 'LineWidth', 2);
hold on
plot(vgg16_val_loss, '-b', 'LineWidth', 2);
grid on
box on
legend({'Training loss', 'Validation loss'});
xlabel('Epoch')
ylabel('Cross entropy loss')
set(gca, 'FontSize', 16, 'FontAngle', 'italic', 'FontWeight', 'bold')

subplot(1,3,2);
plot(resnet50_train_loss, '-r', 'LineWidth', 2);
hold on
plot(resnet50_val_loss, '-b', 'LineWidth', 2);
grid on
box on
legend({'Training loss', 'Validation loss'});
xlabel('Epoch')
ylabel('Cross entropy loss')
set(gca, 'FontSize', 16, 'FontAngle', 'italic', 'FontWeight', 'bold')

subplot(1,3,3);
plot(densenet121_train_loss, '-r', 'LineWidth', 2);
hold on
plot(densenet121_val_loss, '-b', 'LineWidth', 2);
grid on
box on
legend({'Training loss', 'Validation loss'});
xlabel('Epoch')
ylabel('Cross entropy loss')
set(gca, 'FontSize', 16, 'FontAngle', 'italic', 'FontWeight', 'bold')

%% Plot results of training VGG with varying brightness, sharpness, and contrast
figure();
%Start with contrast
baseDir = './experiment_data/VGG16_LR0.25e-3_NoRotFlip_Contrast';
contrastLevels = {'025', '15', '25'};
legends = {'Train loss - Contrast 0.25', 'Val loss - Contrast 0.25', 'Train loss - Contrast 1.5', 'Val loss - Contrast 1.5','Train loss - Contrast 2.5', 'Val loss - Contrast 2.5'};
lineSpec = {'--or', '-^r', '--og', '-^g', '--ob', '-^b'};
markerStartInd = [1 5 10];
subplot(1,3,1);
hold on
j=1;
for k = 1 : 3
    train_loss = table2array(readtable([baseDir contrastLevels{k} '/training_losses.txt'], 'ReadVariableNames', false));
    train_loss{1,1} = train_loss{1,1}(2:end);
    train_loss{1,end} = train_loss{1,end}(1:end-1);
    train_loss = cell2mat(cellfun(@str2double, train_loss, 'UniformOutput', false));
    
    val_loss = table2array(readtable([baseDir contrastLevels{k} '/val_losses.txt'], 'ReadVariableNames', false));
    val_loss{1,1} = val_loss{1,1}(2:end);
    val_loss{1,end} = val_loss{1,end}(1:end-1);
    val_loss = cell2mat(cellfun(@str2double, val_loss, 'UniformOutput', false));
    plot(train_loss, lineSpec{j}, 'LineWidth', 2, 'MarkerIndices', markerStartInd(k):10:100, 'MarkerSize', 10);
    plot(val_loss, lineSpec{j+1}, 'LineWidth', 2, 'MarkerIndices', markerStartInd(k):10:100, 'MarkerSize', 10);
    j = j+2;
end
xlabel('Epoch')
ylabel('Cross entropy loss');
ylim([0 3]);
legend(legends);
box on
grid on
set(gca, 'FontSize', 16, 'FontAngle', 'italic', 'FontWeight', 'bold');


baseDir = './experiment_data/VGG16_LR0.25e-3_NoRotFlip_Brightness';
contrastLevels = {'025', '05', '25'};
legends = {'Train loss - Brightness 0.25', 'Val loss - Brightness 0.25', 'Train loss - Brightness 0.5', 'Val loss - Brightness 0.5','Train loss - Brightness 2.5', 'Val loss - Brightness 2.5'};
lineSpec = {'--or', '-^r', '--og', '-^g', '--ob', '-^b'};
markerStartInd = [1 5 10];
subplot(1,3,2);
hold on
j=1;
for k = 1 : 3
    train_loss = table2array(readtable([baseDir contrastLevels{k} '/training_losses.txt'], 'ReadVariableNames', false));
    train_loss{1,1} = train_loss{1,1}(2:end);
    train_loss{1,end} = train_loss{1,end}(1:end-1);
    train_loss = cell2mat(cellfun(@str2double, train_loss, 'UniformOutput', false));
    
    val_loss = table2array(readtable([baseDir contrastLevels{k} '/val_losses.txt'], 'ReadVariableNames', false));
    val_loss{1,1} = val_loss{1,1}(2:end);
    val_loss{1,end} = val_loss{1,end}(1:end-1);
    val_loss = cell2mat(cellfun(@str2double, val_loss, 'UniformOutput', false));
    plot(train_loss, lineSpec{j}, 'LineWidth', 2, 'MarkerIndices', markerStartInd(k):10:100, 'MarkerSize', 10);
    plot(val_loss, lineSpec{j+1}, 'LineWidth', 2, 'MarkerIndices', markerStartInd(k):10:100, 'MarkerSize', 10);
    j = j+2;
end
xlabel('Epoch')
ylabel('Cross entropy loss');
ylim([0 3]);
legend(legends);
box on
grid on
set(gca, 'FontSize', 16, 'FontAngle', 'italic', 'FontWeight', 'bold');

baseDir = './experiment_data/VGG16_LR0.25e-3_NoRotFlip_Sharpness';
contrastLevels = {'002', '05', '25'};
legends = {'Train loss - Sharpness 0.02', 'Val loss - Sharpness 0.02', 'Train loss - Sharpness 0.5', 'Val loss - Sharpness 0.5','Train loss - Sharpness 2.5', 'Val loss - Sharpness 2.5'};
lineSpec = {'--or', '-^r', '--og', '-^g', '--ob', '-^b'};
markerStartInd = [1 5 10];
subplot(1,3,3);
hold on
j=1;
for k = 1 : 3
    train_loss = table2array(readtable([baseDir contrastLevels{k} '/training_losses.txt'], 'ReadVariableNames', false));
    train_loss{1,1} = train_loss{1,1}(2:end);
    train_loss{1,end} = train_loss{1,end}(1:end-1);
    train_loss = cell2mat(cellfun(@str2double, train_loss, 'UniformOutput', false));
    
    val_loss = table2array(readtable([baseDir contrastLevels{k} '/val_losses.txt'], 'ReadVariableNames', false));
    val_loss{1,1} = val_loss{1,1}(2:end);
    val_loss{1,end} = val_loss{1,end}(1:end-1);
    val_loss = cell2mat(cellfun(@str2double, val_loss, 'UniformOutput', false));
    plot(train_loss, lineSpec{j}, 'LineWidth', 2, 'MarkerIndices', markerStartInd(k):10:100, 'MarkerSize', 10);
    plot(val_loss, lineSpec{j+1}, 'LineWidth', 2, 'MarkerIndices', markerStartInd(k):10:100, 'MarkerSize', 10);
    j = j+2;
end
xlabel('Epoch')
ylabel('Cross entropy loss');
ylim([0 3]);
legend(legends);
box on
grid on
set(gca, 'FontSize', 16, 'FontAngle', 'italic', 'FontWeight', 'bold');