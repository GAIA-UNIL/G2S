source('~/githubProject/G2S/build/R-build/g2s.R')
source=array(runif(10000, 0.0, 1.0),c(100,100));
destination=array(dim=c(100,100))*1;
data=g2s('-a','qs','-ti',source,'-di',destination,'-dt',array(0,c(1,1)),'-k',1.5,'-n',50);

# with image
library(png)
img<-readPNG("../TrainingImages/source.png")
destination=array(dim=c(200,200))*1;
data=g2s('-a','qs','-ti',img,'-di',destination,'-dt',array(0,c(1,1)),'-k',1.5,'-n',50);


