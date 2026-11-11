// Resources page: light / dark switch of each animation. It starts on the system appearance and
// changes both the animated SVG shown and the files the download buttons point to.
(function () {
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  function apply(card, theme, explicit) {
    var img = card.querySelector('picture img');
    if (explicit) {
      // an explicit choice wins over the <source> that follows the system appearance
      var source = card.querySelector('picture source');
      if (source) source.parentNode.removeChild(source);
      img.src = card.dataset[theme + 'Svg'];
    }
    card.querySelector('a[data-kind="svg"]').href = card.dataset[theme + 'Svg'];
    card.querySelector('a[data-kind="mp4"]').href = card.dataset[theme + 'Mp4'];
    card.querySelectorAll('.theme-switch button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.theme === theme));
    });
  }

  document.querySelectorAll('.animation-card').forEach(function (card) {
    var chosen = false;
    var system = function () { return media && media.matches ? 'dark' : 'light'; };
    apply(card, system(), false);
    var sw = card.querySelector('.theme-switch');
    sw.hidden = false;
    sw.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-theme]');
      if (!b) return;
      chosen = true;
      apply(card, b.dataset.theme, true);
    });
    if (media && media.addEventListener) {
      media.addEventListener('change', function () { if (!chosen) apply(card, system(), false); });
    }
  });
})();
