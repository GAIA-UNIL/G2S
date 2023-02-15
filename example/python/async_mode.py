import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt
import time

# load example training image ('stone')
ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)));


# asynchronous QS call using G2S with the "-submitOnly" flag
jobid_1=g2s('-a','qs', 
                 '-submitOnly',
                 '-ti',ti,
                 '-di',numpy.zeros((200,200))*numpy.nan,
                 '-dt',[0],
                 '-k',1.2,
                 '-n',50,
                 '-j',0.5);

# second asynchronous call that waits for job 1 to finish using the "-after" flag
jobid_2 = g2s('-a','qs', 
                  '-after',jobid_1,
                  '-submitOnly',
                  '-ti',ti,
                  '-di',numpy.zeros((200,200))*numpy.nan,
                  '-dt',[0],
                  '-k',1.2,
                  '-n',50,
                  '-j',0.5);

# check the status of both jobs in 2-second intervals using the "-statusOnly" flag
status_1 = 0
status_2 = 0
while status_2 < 95:
    time.sleep(2)
    status_1,*_ = g2s('-statusOnly',jobid_1)
    status_2,*_ = g2s('-statusOnly',jobid_2)
    print('Status jobs 1 & 2:   ', status_1,status_2)

# retrieve the simulation results from the server using the "-waitAndDownload" flag
# if the simulation would not be ready yet this call would wait for it to be ready
sim1,*_ = g2s('-waitAndDownload',jobid_1)
sim2,*_ = g2s('-waitAndDownload',jobid_2)

# display results 
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(7,4))
fig.suptitle('QS Unconditional simulation',size='xx-large')
ax1.imshow(ti)
ax1.set_title('Training image');
ax1.axis('off');
ax2.imshow(sim1)
ax2.set_title('Simulation 1');
ax2.axis('off');
ax3.imshow(sim2)
ax3.set_title('Simulation 2');
ax3.axis('off');
plt.show()
