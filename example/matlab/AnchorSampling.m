% Pronunciation note:
% say AS like the French "as" (ace of a deck), or spell it "A, S".

ti1=ones(21,21);
ti2=2*ones(21,21);
ti1(11,11)=5;
ti2(11,11)=9;

conditioning=nan(21,21,1);
conditioning(10,11)=1;
conditioning(11,10)=1;
conditioning(11,12)=1;
conditioning(12,11)=1;

path=100+reshape(0:numel(conditioning(:,:,1))-1,21,21);
path(11,11)=0;

mask=zeros(21,21,2);
mask(:,:,1)=1;
mask(:,:,2)=0.2;

[simulation,selected_ti]=g2s('-a','as', ...
	'-ti',{ti1 ti2}, ...
	'-di',conditioning, ...
	'-sp',path, ...
	'-mi',mask, ...
	'-dt',[0], ...
	'-k',2, ...
	'-n',8, ...
	'-j',0.5);

figure(1); clf
subplot(1,5,1); imagesc(ti1); axis image off; title('TI 1');
subplot(1,5,2); imagesc(ti2); axis image off; title('TI 2');
subplot(1,5,3); imagesc(conditioning(:,:,1)); axis image off; title('Conditioning');
subplot(1,5,4); imagesc(simulation(:,:,1)); axis image off; title('AS result');
subplot(1,5,5); imagesc(selected_ti(:,:,1)); axis image off; title('Selected TI id');
