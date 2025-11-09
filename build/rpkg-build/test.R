source("./g2s.R")
source=array(runif(10000, 0.0, 1.0),c(100,100));
destination=array(dim=c(100,100))*1;
data=g2s(a='qs',ti=source,di=destination,dt=array(0,c(1,1)),k=1.5,n=50);
image(data[[1]]);

# with image
library(png)
img<-readPNG("../TrainingImages/source.png")
destination=array(dim=c(200,200))*1;
data=g2s(a='qs',ti=img,di=destination,dt=array(0,c(1)),k=1.5,n=50);
image(data[[1]]);


img3b=array(dim=c(200,200,3));
img3b[,,1]=img;
img3b[,,2]=img;
img3b[,,3]=img;

destination3=array(dim=c(200,200,3))*1;
data=g2s(a='qs',ti=img3b,di=destination3,dt=array(0,c(3)),k=1.5,n=50);
image(data[[1]][,,1]);


