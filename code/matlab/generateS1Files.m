function [data] = generateS1Files(directoryOfSamples, outputFile)
   addpath(directoryOfSamples);
   filesPath = strcat(directoryOfSamples,'*.wav');
   files = dir(filesPath);
   numOfFiles = length(files);
   dataTable = table();
   for i = 1:numOfFiles
       audio = files(i).name(1:end-4);
       seg = generateSegmentedDataset(directoryOfSamples, audio);
       s1 = seg(1,1);
       newData = table([audio],[s1]);
       dataTable = [dataTable;newData]
   end
   writetable(dataTable,outputFile);
