function openTab(evt, tabName, type) {
  var i, tabcontent, tablinks, activeTabcontent, allButton;
  tabcontent = document.getElementsByClassName("langcontent "+type);
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].classList.add("hideClass");
  }
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].classList.remove("active");
  }
  console.log(tabName)
  activeTabcontent=document.getElementsByClassName("langcontent "+tabName)
  
  for (i = 0; i < activeTabcontent.length; i++) {
    activeTabcontent[i].classList.remove("hideClass");
  }
  allButton=document.getElementsByClassName("tablinks "+tabName);
  console.log(allButton)
  for (i = 0; i < activeTabcontent.length; i++) {
    allButton[i].classList.add("active");
  }
}

document.addEventListener('DOMContentLoaded', function() {
  openTab(null, "python", "code interface");
});
