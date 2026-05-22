(function() {
  function toggleMore(el) {
    var li = el.closest('.post-item');
    var full = li.querySelector('.post-full-content');
    var excerpt = li.querySelector('.post-excerpt');
    el.blur();
    if (!full.classList.contains('expanded')) {
      excerpt.style.maxHeight = excerpt.scrollHeight + 'px';
      requestAnimationFrame(function() { excerpt.classList.add('collapsed'); });
      full.style.maxHeight = full.scrollHeight + 'px';
      full.classList.add('expanded');
      el.innerHTML = 'Less 🍁';
      full.addEventListener('transitionend', function handler() {
        if (full.classList.contains('expanded')) {
          full.style.overflow = 'visible';
          full.style.maxHeight = 'none';
        }
        full.removeEventListener('transitionend', handler);
      });
    } else {
      full.style.overflow = 'hidden';
      full.style.maxHeight = full.scrollHeight + 'px';
      requestAnimationFrame(function() {
        full.style.maxHeight = '0';
        full.classList.remove('expanded');
      });
      excerpt.classList.remove('collapsed');
      excerpt.style.maxHeight = excerpt.scrollHeight + 'px';
      el.innerHTML = 'More 🌱';
      var headerH = document.querySelector('header').offsetHeight;
      var targetY = li.getBoundingClientRect().top + window.pageYOffset - headerH - 10;
      if ('scrollBehavior' in document.documentElement.style) {
        window.scrollTo({ top: targetY, behavior: 'smooth' });
      } else {
        window.scrollTo(0, targetY);
      }
    }
  }

  document.querySelectorAll('.more-toggle').forEach(function(el) {
    el.addEventListener('click', function(e) {
      e.preventDefault();
      toggleMore(el);
    });
  });

  if (window.Pagination) {
    var hashMatch = location.hash.match(/^#page=(\d+)/);
    var initialPage = hashMatch ? parseInt(hashMatch[1], 10) : 1;
    if (!(initialPage > 1)) initialPage = 1;
    var pageInitialized = false;

    var pager = window.Pagination.init({
      items: document.querySelectorAll('.post-item'),
      perPage: 5,
      paginationEl: document.getElementById('pagination'),
      onPageChange: function(page) {
        if (pageInitialized) {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        var newUrl = page > 1
          ? location.pathname + location.search + '#page=' + page
          : location.pathname + location.search;
        if (history.replaceState) history.replaceState(null, '', newUrl);
      }
    });

    if (initialPage > 1 && pager) {
      pager.goPage(initialPage);
    }
    pageInitialized = true;
  }
})();
