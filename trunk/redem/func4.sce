j=1;
yy=0;

//12 [0.15:0.07:0.98];
//12 [-0.98:0.07:-0.15];
//76 [-0.15:0.004:-015];

x = [-0.98:0.07:-0.09];
x = [x [-0.08:0.002:0.08]];
x = [x [0.09:0.07:1]];

for k=1:length(x)
  yy(j) = abs(x(k));
  j = j + 1;
end

fprintfMat('entrada108.txt',x','%.3f');
fprintfMat('desejada108.txt',yy,'%.3f');
plot2d(x',yy);



