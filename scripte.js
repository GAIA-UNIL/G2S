$(document).ready(function(){
	/*$("div.quickSelect").each(function(){
		$(this).css("max-height", $(this).height());
		//.addClass("height_"+Math.ceil($(this).height()/10)*10);
	});*/
	//OS
	$("div.osChoice a.button:not([disabled])").click(function(){
		$("div.osChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.osChoice a.button."+button.data('os')).attr('selected', true);
		$("div.quickSelect.osSens."+button.data('os')).removeClass("os2hide");
		$("div.quickSelect.osSens:not('."+button.data('os')+"')").addClass("os2hide");
		if(event) event.stopPropagation();
	});
	//server / inteface
	$("div.installationChoice a.button:not([disabled])").click(function(){
		$("div.installationChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.installationChoice a.button."+button.data('install')).attr('selected', true);
		$("div.quickSelect.installSens."+button.data('install')).removeClass("install2hide");
		$("div.quickSelect.installSens:not('."+button.data('install')+"')").addClass("install2hide");
		if(event) event.stopPropagation();
	});

	$("div.progLangChoice a.button:not([disabled])").click(function(){
		$("div.progLangChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.progLangChoice a.button."+button.data('proglang')).attr('selected', true);
		$("div.quickSelect.langSens."+button.data('proglang')).removeClass("progLang2hide");
		$("div.quickSelect.langSens:not('."+button.data('proglang')+"')").addClass("progLang2hide");
		if(event) event.stopPropagation();
	});

	$("div.algoChoice a.button:not([disabled])").click(function(){
		$("div.algoChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.algoChoice a.button."+button.data('algo')).attr('selected', true);
		$("div.quickSelect.algoSens."+button.data('algo')).removeClass("algo2hide");
		$("div.quickSelect.algoSens:not('."+button.data('algo')+"')").addClass("algo2hide");
		
		if(event) event.stopPropagation();
		if(autoScrollOnClick){
			var nativeButton=this;
			setTimeout(function(){
				let newPositionInView=nativeButton.getBoundingClientRect();
				if(newPositionInView.top<0 || $(window).height()){
					nativeButton.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
				}		
			},100);
		}
	});

	$("div.exampleChoice a.button:not([disabled])").click(function(){
		$("div.exampleChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.exampleChoice a.button."+button.data('example')).attr('selected', true);
		$("div.quickSelect.exampleSens."+button.data('example')).removeClass("example2hide");
		$("div.quickSelect.exampleSens:not('."+button.data('example')+"')").addClass("example2hide");
		if(event) event.stopPropagation();
	});

	$('<th class="copyCell"></td>').insertBefore($('table thead tr th:first-child'))

	$("td.addCopy").each(function(){
		$('<td><i class="fas fa-copy"></i></td>').click(function(){
			console.log($(this).next().html())
			navigator.clipboard.writeText($(this).next().html().replace(String.fromCharCode(8209), '-').replace("&nbsp;", ' '));
		}).insertBefore(this);
	});

	// click(function(){
	// 	$("div.exampleChoice a.button").attr('selected', false);
	// 	var button=$(this);
	// 	$("div.exampleChoice a.button."+button.data('example')).attr('selected', true);
	// 	$("div.quickSelect.exampleSens."+button.data('example')).removeClass("example2hide");
	// 	$("div.quickSelect.exampleSens:not('."+button.data('example')+"')").addClass("example2hide");
	// 	if(event) event.stopPropagation();
	// });

	autoSetOs();
	autoScrollOnClick=false;
	autoSetAlgo();
	autoScrollOnClick=true;
	loadAllExmaples();

});

function setMaxHeight(parent){
	$(parent).css("max-height", $(parent).height());
}

function loadAllExmaples(){
	$('code[src]').each(function(idx,el){
		$.ajax($(el).attr('src')).done(function(data){
			$(el).html(data)
			hljs.highlightElement(el);
			var parentObject=$(el).closest('div.quickSelect.exampleSens');
			setMaxHeight(parentObject)
		})
	})
}

function autoSetOs() {
	var userAgent = window.navigator.userAgent,
	platform = window.navigator.platform,
	macosPlatforms = ['Macintosh', 'MacIntel', 'MacPPC', 'Mac68K','darwin'],
	windowsPlatforms = ['Win32', 'Win64', 'Windows', 'WinCE'],
	iosPlatforms = ['iPhone', 'iPad', 'iPod'],
	os=null;

	if (macosPlatforms.indexOf(platform) !== -1) {
		os = 'macOS';
	} else if (windowsPlatforms.indexOf(platform) !== -1) {
		os = 'windows';
	} else if (!os && /Linux/.test(platform)) {
		os = 'inux';
	}

	if(os){
		$("div.osChoice.buttonChoice a.button."+os).click();
	}
}

function autoSetAlgo() {
	//automatically start with QS
	$("div.algoChoice.buttonChoice a.button.qs").click();
}