document.addEventListener('DOMContentLoaded', function() {
  var backToTop = document.getElementById('backToTop');
  if (!backToTop) return;
  var ticking = false;
  window.addEventListener('scroll', function() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function() {
      if (window.pageYOffset > 100) {
        backToTop.classList.add('show');
      } else {
        backToTop.classList.remove('show');
      }
      ticking = false;
    });
  }, { passive: true });
  backToTop.addEventListener('click', function(e) {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});
