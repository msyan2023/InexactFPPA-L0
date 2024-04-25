function [TrainData, TestData, TrainLabel, TestLabel] = ReadData(FileName)
% format long
switch FileName
    case 'Housing'
        %% Housing data
        FileTxT = fopen('Housing.txt');
        cell_data= textscan(FileTxT,'%s%s%s%s%s%s%s%s%s%s%s%s%s%s','Delimiter',' ');
        NumTrain = 300;
    case 'Mg'
        %% Mg data
        FileTxT = fopen('Mg.txt');
        cell_data= textscan(FileTxT,'%s%s%s%s%s%s%s','Delimiter',' ');
        NumTrain = 1000;
end
CellData = cat(2,cell_data{:});
fclose(FileTxT);


Data = cellfun(@(x) extractAfter(x,":"),CellData(:,2:end),'UniformOutput',false);
Data = cellfun(@str2double,Data);
Data = Data.';

Label = cellfun(@str2double, CellData(:,1));


TrainData = Data(:,1:NumTrain);
TrainLabel = Label(1:NumTrain);
TestData = Data(:,NumTrain+1:end);
TestLabel = Label(NumTrain+1:end);

end