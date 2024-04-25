function [Data, Label] = ReadData(FileName)
% format long
switch FileName
    case 'Housing'
        %% Housing data
        FileTxT = fopen('Housing.txt');
        cell_data= textscan(FileTxT,'%s%s%s%s%s%s%s%s%s%s%s%s%s%s','Delimiter',' ');
    case 'Mg'
        %% Mg data
        FileTxT = fopen('Mg.txt');
        cell_data= textscan(FileTxT,'%s%s%s%s%s%s%s','Delimiter',' ');
end
CellData = cat(2,cell_data{:});
fclose(FileTxT);


Data = cellfun(@(x) extractAfter(x,":"),CellData(:,2:end),'UniformOutput',false);
Data = cellfun(@str2double,Data);
Data = Data.';

Label = cellfun(@str2double, CellData(:,1));



end