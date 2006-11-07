j=1;
yy=0;

for x=-0.98:0.02:1
//  if x < 0.5
//    yy(j) = 1 - abs(x);
    yy(j) = abs(x);
//  elseif x >= 0.5
//    yy(j) = 1 - abs(x);
//  end
  j = j + 1;
end

fprintfMat('entrada100.txt',[-0.98:0.02:1]','%.2f');
fprintfMat('desejada100.txt',yy,'%.2f');
plot2d([-0.98:0.02:1]',yy);


