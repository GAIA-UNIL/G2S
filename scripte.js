$(document).ready(function(){
	$("div.quickSelect").each(function(){
		$(this).css("max-height", $(this).height());
		//.addClass("height_"+Math.ceil($(this).height()/10)*10);
	});
	//OS
	$("div.osChoice a.button:not([disabled])").click(function(){
		$("div.osChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.osChoice a.button."+button.data('os')).attr('selected', true);
		$("div.quickSelect.osSens."+button.data('os')).removeClass("os2hide");
		$("div.quickSelect.osSens:not('."+button.data('os')+"')").addClass("os2hide");
		event.stopPropagation();
	});
	//server / inteface
	$("div.installationChoice a.button:not([disabled])").click(function(){
		$("div.installationChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.installationChoice a.button."+button.data('install')).attr('selected', true);
		$("div.quickSelect.installSens."+button.data('install')).removeClass("install2hide");
		$("div.quickSelect.installSens:not('."+button.data('install')+"')").addClass("install2hide");
		event.stopPropagation();
	});

	$("div.progLangChoice a.button:not([disabled])").click(function(){
		$("div.progLangChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.progLangChoice a.button."+button.data('proglang')).attr('selected', true);
		$("div.quickSelect.langSens."+button.data('proglang')).removeClass("progLang2hide");
		$("div.quickSelect.langSens:not('."+button.data('proglang')+"')").addClass("progLang2hide");
		event.stopPropagation();
	});

	$("div.algoChoice a.button:not([disabled])").click(function(){
		$("div.algoChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.algoChoice a.button."+button.data('algo')).attr('selected', true);
		$("div.quickSelect.algoSens."+button.data('algo')).removeClass("algo2hide");
		$("div.quickSelect.algoSens:not('."+button.data('algo')+"')").addClass("algo2hide");
		event.stopPropagation();
	});

	$("div.exampleChoice a.button:not([disabled])").click(function(){
		$("div.exampleChoice a.button").attr('selected', false);
		var button=$(this);
		$("div.exampleChoice a.button."+button.data('example')).attr('selected', true);
		$("div.quickSelect.exampleSens."+button.data('example')).removeClass("example2hide");
		$("div.quickSelect.exampleSens:not('."+button.data('example')+"')").addClass("example2hide");
		event.stopPropagation();
	});
});

