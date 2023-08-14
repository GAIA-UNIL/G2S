% load example traning image ('stone')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');

% asynchronous QS call using G2S with the "-submitOnly" flag
jobid_1=g2s('-a','qs','-submitOnly','-ti',ti,'-di',nan(200,200),'-dt',[0],'-k',1.2,'-n',50,'-j',0.5);

% 2nd async call that waits for job 1 to finish using the "-after" flag
jobid_2=g2s('-a','qs','-after',jobid_1,'-submitOnly','-ti',ti,'-di',nan(200,200),'-dt',[0],'-k',1.2,'-n',50,'-j',0.5);

% check the status of both jobs in 2-second intervals using the
% "-statusOnly" flag
status_1 = 0;
status_2 = 0;
while status_2 < 95
    pause(2);
    status_1=g2s('-statusOnly',jobid_1);
    status_2=g2s('-statusOnly',jobid_2);
    fprintf('Status jobs 1 & 2:   %s %s\n', status_1, status_2);
end

% retrieve the simulation results from the server using the
%"-waitAndDownload" flag. If the simulation would not be ready yet this
%call would wait for it to be ready
sim1=g2s('-waitAndDownload',jobid_1);
sim2=g2s('-waitAndDownload',jobid_2);

% display results
sgtitle('Unconditional asynchronous simulation');
subplot(1,3,1);
imshow(ti);
title('Training image');
subplot(1,3,2);
imshow(sim1);
title('Simulation 1');
subplot(1,3,3);
imshow(sim2);
title('Simulation 2');


