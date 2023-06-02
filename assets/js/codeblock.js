document.addEventListener('DOMContentLoaded', function() {
  var codeBlocks = document.querySelectorAll('pre.highlight');

  codeBlocks.forEach(function (codeBlock) {
    var copyButton = document.createElement('span');
    copyButton.classList.add("material-symbols-outlined","copy");
    copyButton.innerText = 'content_copy';

    codeBlock.parentElement.prepend(copyButton);

    copyButton.addEventListener('click', function () {
      var code = codeBlock.querySelector('code').innerText.trim();
      window.navigator.clipboard.writeText(code);
    });
  });
});