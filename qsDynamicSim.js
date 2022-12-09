targetSimProgression=0.;
var qsIsPause=false;
const viridis=[[253, 231, 37],[251, 231, 35],[248, 230, 33],[246, 230, 32],[244, 230, 30],[241, 229, 29],[239, 229, 28],[236, 229, 27],[234, 229, 26],[231, 228, 25],[229, 228, 25],[226, 228, 24],[223, 227, 24],[221, 227, 24],[218, 227, 25],[216, 226, 25],[213, 226, 26],[210, 226, 27],[208, 225, 28],[205, 225, 29],[202, 225, 31],[200, 224, 32],[197, 224, 33],[194, 223, 35],[192, 223, 37],[189, 223, 38],[186, 222, 40],[184, 222, 41],[181, 222, 43],[178, 221, 45],[176, 221, 47],[173, 220, 48],[170, 220, 50],[168, 219, 52],[165, 219, 54],[162, 218, 55],[160, 218, 57],[157, 217, 59],[155, 217, 60],[152, 216, 62],[149, 216, 64],[147, 215, 65],[144, 215, 67],[142, 214, 69],[139, 214, 70],[137, 213, 72],[134, 213, 73],[132, 212, 75],[129, 211, 77],[127, 211, 78],[124, 210, 80],[122, 209, 81],[119, 209, 83],[117, 208, 84],[115, 208, 86],[112, 207, 87],[110, 206, 88],[108, 205, 90],[105, 205, 91],[103, 204, 92],[101, 203, 94],[99, 203, 95],[96, 202, 96],[94, 201, 98],[92, 200, 99],[90, 200, 100],[88, 199, 101],[86, 198, 103],[84, 197, 104],[82, 197, 105],[80, 196, 106],[78, 195, 107],[76, 194, 108],[74, 193, 109],[72, 193, 110],[70, 192, 111],[68, 191, 112],[66, 190, 113],[64, 189, 114],[63, 188, 115],[61, 188, 116],[59, 187, 117],[58, 186, 118],[56, 185, 119],[55, 184, 120],[53, 183, 121],[52, 182, 121],[50, 182, 122],[49, 181, 123],[47, 180, 124],[46, 179, 124],[45, 178, 125],[44, 177, 126],[42, 176, 127],[41, 175, 127],[40, 174, 128],[39, 173, 129],[38, 173, 129],[37, 172, 130],[37, 171, 130],[36, 170, 131],[35, 169, 131],[34, 168, 132],[34, 167, 133],[33, 166, 133],[33, 165, 133],[32, 164, 134],[32, 163, 134],[31, 162, 135],[31, 161, 135],[31, 161, 136],[31, 160, 136],[31, 159, 136],[31, 158, 137],[30, 157, 137],[30, 156, 137],[30, 155, 138],[31, 154, 138],[31, 153, 138],[31, 152, 139],[31, 151, 139],[31, 150, 139],[31, 149, 139],[31, 148, 140],[32, 147, 140],[32, 146, 140],[32, 146, 140],[33, 145, 140],[33, 143, 141],[33, 142, 141],[34, 141, 141],[34, 140, 141],[34, 139, 141],[35, 138, 141],[35, 137, 142],[35, 136, 142],[36, 135, 142],[36, 134, 142],[37, 133, 142],[37, 132, 142],[37, 131, 142],[38, 130, 142],[38, 130, 142],[38, 129, 142],[39, 128, 142],[39, 127, 142],[39, 126, 142],[40, 125, 142],[40, 124, 142],[41, 123, 142],[41, 122, 142],[41, 121, 142],[42, 120, 142],[42, 119, 142],[42, 118, 142],[43, 117, 142],[43, 116, 142],[44, 115, 142],[44, 114, 142],[44, 113, 142],[45, 113, 142],[45, 112, 142],[46, 111, 142],[46, 110, 142],[46, 109, 142],[47, 108, 142],[47, 107, 142],[48, 106, 142],[48, 105, 142],[49, 104, 142],[49, 103, 142],[49, 102, 142],[50, 101, 142],[50, 100, 142],[51, 99, 141],[51, 98, 141],[52, 97, 141],[52, 96, 141],[53, 95, 141],[53, 94, 141],[54, 93, 141],[54, 92, 141],[55, 91, 141],[55, 90, 140],[56, 89, 140],[56, 88, 140],[57, 86, 140],[57, 85, 140],[58, 84, 140],[58, 83, 139],[59, 82, 139],[59, 81, 139],[60, 80, 139],[60, 79, 138],[61, 78, 138],[61, 77, 138],[62, 76, 138],[62, 74, 137],[62, 73, 137],[63, 72, 137],[63, 71, 136],[64, 70, 136],[64, 69, 136],[65, 68, 135],[65, 66, 135],[66, 65, 134],[66, 64, 134],[66, 63, 133],[67, 62, 133],[67, 61, 132],[68, 59, 132],[68, 58, 131],[68, 57, 131],[69, 56, 130],[69, 55, 129],[69, 53, 129],[70, 52, 128],[70, 51, 127],[70, 50, 126],[70, 48, 126],[71, 47, 125],[71, 46, 124],[71, 45, 123],[71, 44, 122],[71, 42, 122],[72, 41, 121],[72, 40, 120],[72, 38, 119],[72, 37, 118],[72, 36, 117],[72, 35, 116],[72, 33, 115],[72, 32, 113],[72, 31, 112],[72, 29, 111],[72, 28, 110],[72, 27, 109],[72, 26, 108],[72, 24, 106],[72, 23, 105],[72, 22, 104],[72, 20, 103],[71, 19, 101],[71, 17, 100],[71, 16, 99],[71, 14, 97],[71, 13, 96],[70, 11, 94],[70, 10, 93],[70, 8, 92],[70, 7, 90],[69, 5, 89],[69, 4, 87],[68, 2, 86],[68, 1, 84]]

var speedup=1;

var n=50;
var k=4;

function setSimValues(){
	k=document.querySelector('#k').value
	n=document.querySelector('#n').value

	document.querySelector('#kValue').setHTML(k)
	document.querySelector('#nValue').setHTML(n)
}

document.addEventListener('DOMContentLoaded',function(){
	setSimValues();
	document.querySelector('#n').addEventListener('change',setSimValues);
	document.querySelector('#k').addEventListener('change',setSimValues);
})

function shuffleArray(array) {
  let currentIndex = array.length,  randomIndex;

  // While there remain elements to shuffle.
  while (currentIndex != 0) {

    // Pick a remaining element.
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }

  return array;
}

function rgbToHex(r, g, b) {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function grayToHex(val) {
  return rgbToHex(val,val,val);
}

function getLocationSample(im,sample,sampleVector,k){
	for (var i = 0; i < im.cm.length; i++) {
		im.cm[i]=Infinity;
	}

	minVal=[0,0];
	maxVal=[0,0];

	for (var i = 0; i < sampleVector.length; i++) {
		if(sampleVector[i][0]<minVal[0]) minVal[0]=sampleVector[i][0];
		if(sampleVector[i][1]<minVal[1]) minVal[1]=sampleVector[i][1];
		if(sampleVector[i][0]>maxVal[0]) maxVal[0]=sampleVector[i][0];
		if(sampleVector[i][1]>maxVal[1]) maxVal[1]=sampleVector[i][1];
	}

	for (var y = -minVal[1]; y < im.size[1]-maxVal[1]; y++) {
		for (var x = -minVal[0]; x < im.size[0]-maxVal[0]; x++) {
			let error=0;
			for (var i = 0; i < sample.length; i++) {
				let xloc=sampleVector[i][0]+x;
				let yloc=sampleVector[i][1]+y;
				let val=im.data[xloc+(yloc)*im.size[0]];
				let diff=val-sample[i];
				error+=diff*diff;
			}
			im.cm[x+(y)*im.size[0]]=error+Math.random()*0.00001;
		}
	}

	let pos=[];
	let val=[]
	for (var i = 0; i < k; i++) {
		let min=Math.min(...im.cm);
		val.push(min);
		let loc=im.cm.indexOf(min);
		pos.push(loc)
		im.cm[loc]=Infinity;
	}

	for (var i = pos.length - 1; i >= 0; i--) {
		im.cm[pos[i]]=val[i];
	}

	return pos;

}


function loadQSannim(im){

	maxRad=25;

	let sim = Snap("#sim");
	let ti = Snap("#ti");

	simSize=[50,50];

	let tiLinearSize=im.size[0]*im.size[1];
	let simLinearSize=simSize[0]*simSize[1];
	let simArray=new Array(simLinearSize).map(e=>NaN)
	im.cm=new Array(tiLinearSize);

	let ticanvas = document.createElement('canvas');
  ticanvas.width = im.size[0];
  ticanvas.height = im.size[1];
  let ticanvasCtx=ticanvas.getContext("2d");
  const ti_imageData = ticanvasCtx.getImageData(0, 0, im.size[0],im.size[1]);

  let tierrorcanvas = document.createElement('canvas');
  tierrorcanvas.width = im.size[0];
  tierrorcanvas.height = im.size[1];
  let tierrorcanvasCtx=tierrorcanvas.getContext("2d");
  const tierror_imageData = tierrorcanvasCtx.getImageData(0, 0, im.size[0],im.size[1]);

  let simcanvas = document.createElement('canvas');
  simcanvas.width = simSize[0];
  simcanvas.height = simSize[1];
  let simcanvasCtx=simcanvas.getContext("2d");
  const sim_imageData = simcanvasCtx.getImageData(0, 0, simSize[0],simSize[1]);
	
	for (let y = 0; y < im.size[1]; y++) {
		for (var x = 0; x < im.size[0]; x++) {
			let colorPix=im.data[x+y*im.size[0]]*255;
			for (var i = 0; i < 3; i++) {
				ti_imageData.data[(x+y*im.size[0])*4+i]=colorPix;
			}
			ti_imageData.data[(x+y*im.size[0])*4+3]=255;
		}
	}


	for (var x = 0; x < simLinearSize; x++) {
		sim_imageData.data[x*4+0]=255;
		sim_imageData.data[x*4+1]=0;
		sim_imageData.data[x*4+2]=255;
		sim_imageData.data[x*4+3]=50;
	}

	for (var x = 0; x < tiLinearSize; x++) {
		tierror_imageData.data[x*4+3]=0;
	}




	ticanvasCtx.putImageData(ti_imageData, 0, 0);
	tierrorcanvasCtx.putImageData(tierror_imageData, 0, 0);
	simcanvasCtx.putImageData(sim_imageData, 0, 0);
	tiim=ti.image(ticanvas.toDataURL('image/png'),0,0,1,1)
	tierrorOverlay=ti.image(tierrorcanvas.toDataURL('image/png'),0,0,1,1)
	simim=sim.image(simcanvas.toDataURL('image/png'),0,0,1,1).attr('image-rendering',"pixelated");

	let path=shuffleArray([...Array(simLinearSize).keys()])
	let position=0;

	let searchPath=[... new Array((2*maxRad+1)*(2*maxRad+1))]
		.map((e,i)=>[i%(2*maxRad+1)-maxRad,Math.floor(i/(2*maxRad+1))-maxRad])
		.sort((a,b)=> (a[0]*a[0]+a[1]*a[1])-(b[0]*b[0]+b[1]*b[1]))

	selectedSample=[];

	for (var i = 0; i < 4; i++) {
		let sampleVisu=ti.rect(0, 0, 5/im.size[0], 5/im.size[1], 1/im.size[0],1/im.size[1])
		sampleVisu.attr('opacity',0);
		sampleVisu.attr({
			stroke: "#FF0000",
	    	strokeWidth: 0.01
	    });
		selectedSample.push(sampleVisu)
	}

	let neihboursArray=[];

	for (var i = 0; i < 50; i++) {
		let sampleVisu=sim.circle(0, 0, 1/simSize[0])
		//sampleVisu.attr('opacity',0);
		sampleVisu.attr({
			stroke: "#FF0000",
    	strokeWidth: 0.01,
    	fill:"#0000",
    	opacity:0
    });
		neihboursArray.push(sampleVisu)
	}

	let selectorArrow=ti.polygon(-0.05,1.05,0.05,1.05,0,1.12).attr({ fill: "green"})
	let simulatedCenter=sim.circle(0, 0, 1/simSize[0]).attr({stroke: "#00FF00",
    	strokeWidth: 0.01,fill:'#0000','stroke-opacity': 0})

	function simIfNeeded(){
		let loadPromises=[];
		selectorArrow.animate({
				'opacity':.0,
				'transform':'t0.1,0'
			},1)

		for (var i = 50 - 1; i >= 0; i--) {
			neihboursArray[i].attr({'opacity':1,
							'stroke-opacity': 0
						});
		}
		targetPosition=Math.ceil(simLinearSize*targetSimProgression);
		if(targetPosition==position || qsIsPause)
		{
			setTimeout(simIfNeeded,100);
			return
		}

		speedup=Math.min(Math.log(Math.abs(targetPosition-position)+1)+1,10);

		if(targetPosition<position){
			position--;
			let currentPosIndex=path[position];
			simArray[currentPosIndex]=NaN;
			sim_imageData.data[currentPosIndex*4+0]=255;
			sim_imageData.data[currentPosIndex*4+1]=0;
			sim_imageData.data[currentPosIndex*4+2]=255;
			sim_imageData.data[currentPosIndex*4+3]=50;
			let y=Math.floor(currentPosIndex/simSize[0]);
			let x=currentPosIndex%simSize[0];
			simulatedCenter.attr({
				'stroke-opacity': 1,
				'transform':'t'+(x/simSize[0]+0.5/simSize[0])+','+(y/simSize[1]+0.5/simSize[0])+',s1',
			})

			simcanvasCtx.putImageData(sim_imageData, 0, 0);
			simim.attr('xlink:href',simcanvas.toDataURL('image/png'));
			setTimeout(simIfNeeded,Math.ceil(1000/speedup));	
		}else{
			let currentPosIndex=path[position];
			position++;
			let y=Math.floor(currentPosIndex/simSize[0]);
			let x=currentPosIndex%simSize[0];
			simulatedCenter.attr({
				'stroke-opacity': 1,
				'transform':'t'+(x/simSize[0]+0.5/simSize[0])+','+(y/simSize[1]+0.5/simSize[0])+',s1',
			})
			//search neighbours
			let sample=[];
			let sampleVector=[];

			let p=0;
			let loadPromises=[];
			while((sample.length<n) && (p<searchPath.length))
			{
				let xloc=searchPath[p][0]+x;
				let yloc=searchPath[p][1]+y;
				let val;
				if ((xloc>=0) && (xloc<simSize[0]) && (yloc>=0) && (yloc<simSize[1]) && (val=simArray[xloc+(yloc)*simSize[0]]) && (!isNaN(val))){
					sample.push(val)
					sampleVector.push(searchPath[p])
					neihboursArray[sample.length-1].attr({'opacity':1,
						//'x':0.5*50,//x/simSize[0]-1/simSize[0],
						//'y':0.5*50,//y/simSize[1]-1/simSize[0],
						'stroke-opacity': 0,
						'transform':'t'+(x/simSize[0]+0.5/simSize[0])+','+(y/simSize[1]+0.5/simSize[0])+',s1',
						'fill':'#0000'
					});
					loadPromises.push(
						new Promise(function(resolve, reject) {
					    neihboursArray[sample.length-1].animate({
					    	'transform':'t'+(xloc/simSize[0]+0.5/simSize[0])+','+(yloc/simSize[1]+0.5/simSize[0])+',s1',
								'stroke-opacity': 1,
					    }, Math.ceil(500/speedup),null,resolve);
					}));
				}
				p++;
			}

			let samples=getLocationSample(im,sample,sampleVector,Math.ceil(k));
			for (var j = 0; j < tiLinearSize; j++) {
				tierror_imageData.data[j*4+3]=0;
			}

			let min=Math.min(...im.cm);
			let max=Math.max(...im.cm.filter(e=> e!==Infinity));
			for (var j = 0; j < tiLinearSize; j++) {
				if(im.cm[j]>1e9) continue;
				let cmVal=Math.floor((im.cm[j]-min)/(max-min)*254);
				tierror_imageData.data[j*4+0]=viridis[cmVal][0];
				tierror_imageData.data[j*4+1]=viridis[cmVal][1];
				tierror_imageData.data[j*4+2]=viridis[cmVal][2];
				tierror_imageData.data[j*4+3]=100;
			}

			tierrorcanvasCtx.putImageData(tierror_imageData, 0, 0);
			tierrorOverlay.attr({'xlink:href':tierrorcanvas.toDataURL('image/png'),
				'opacity':0});

			tierrorOverlay.animate({'opacity':1},Math.ceil(600/speedup));
			
			Promise.all(loadPromises).then(function(){
				loadPromises=[];
				for (var i = 0; i < Math.min(selectedSample.length,samples.length); i++) {
					let x=(samples[i]%im.size[0])/im.size[0];
					let y=Math.floor(samples[i]/im.size[0])/im.size[1];
					let size=5/im.size[0];
					selectedSample[i].attr({'opacity':1,
						'x':x,
						'y':y,
						'width':size,
						'height':size,
						'stroke-opacity': 1,
						'transform':'t0,0,s1',
						'fill':grayToHex(Math.floor(im.data[samples[i]]*255))
					});
					
					loadPromises.push(
						new Promise(function(resolve, reject) {
					    selectedSample[i].animate({
					    	'transform': 't'+(-x+0.2*i+0.1+0.1)+','+(1.2-y)+'s'+0.2/size,
							'stroke-opacity': 0,
					    }, Math.ceil(1000/speedup),null,resolve);
					}));

				}

				for (var i =  Math.min(selectedSample.length,samples.length); i< 4; i++) {
					selectedSample[i].attr({'opacity':0,
						'x':0,
						'y':0,
						'width':0,
						'height':0,
						'stroke-opacity': 1,
						'transform':'t-100,-100,s1',
						'fill':'000000'
					});
					
				}

				let selected=k*Math.random();
				Promise.all(loadPromises).then(function(){
					loadPromises=[];
					loadPromises.push(new Promise(function(resolve, reject) {
					    selectorArrow.animate({
							'opacity':1,
							'transform':'t'+(0.2*selected+0.1)+',0'
					    }, Math.ceil(500/speedup),mina.elastic,resolve);
					}));
				});
				

				
				let importIndex=samples[Math.floor(selected)];
				simArray[currentPosIndex]=im.data[importIndex];
				for (var i = 0; i < 3; i++) {
					sim_imageData.data[currentPosIndex*4+i]=simArray[currentPosIndex]*255;
				}
				sim_imageData.data[currentPosIndex*4+3]=255;

				simcanvasCtx.putImageData(sim_imageData, 0, 0);
				simim.attr('xlink:href',simcanvas.toDataURL('image/png'));

				Promise.all(loadPromises).then(function(){
					setTimeout(simIfNeeded,Math.ceil(1000/speedup));	
				})
			});
		}
		
	}
	setTimeout(simIfNeeded,1);
}

document.addEventListener('scroll', (event) => {
	let x=document.documentElement.scrollTop;
	let y=1000;
	if(document.getElementById('Benchmarking'))
		y=document.getElementById('Benchmarking').getBoundingClientRect().top;
	targetSimProgression=x/(x+y);
});

let tiFile=["./stone.json"] //,"./strebelle.json"

fetch(tiFile[Math.floor(tiFile.length*Math.random())])
.then(response => {
   return response.json();
})
.then(jsondata => loadQSannim(jsondata));