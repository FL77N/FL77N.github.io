(function() {
  var searchInput = document.getElementById('search-input');
  var searchResults = document.getElementById('search-results');
  if (!searchInput || !searchResults) return;

  var searchUrl = searchInput.getAttribute('data-search-url') || '/search.json';
  var posts = null;

  function loadPosts(callback) {
    if (posts) { callback(); return; }
    fetch(searchUrl).then(function(res) {
      return res.ok ? res.json() : Promise.reject(res.status);
    }).then(function(data) {
      posts = data;
      callback();
    }).catch(function() { /* swallow */ });
  }

  function doSearch(query) {
    if (!query || query.length < 1) {
      searchResults.style.display = 'none';
      return;
    }
    loadPosts(function() {
      var q = query.toLowerCase();
      var matches = [];
      var seen = {};
      for (var i = 0; i < posts.length; i++) {
        var p = posts[i];
        if (seen[p.url]) continue;
        if (p.title.toLowerCase().indexOf(q) !== -1) {
          matches.push(p);
          seen[p.url] = true;
        }
      }
      if (matches.length === 0) {
        searchResults.innerHTML = '<div class="search-no-results">No results found.</div>';
      } else {
        var html = '';
        for (var j = 0; j < matches.length; j++) {
          html += '<a class="search-result-item" href="' + matches[j].url + '">' +
                  '<span class="search-result-title">' + matches[j].title + '</span>' +
                  '<span class="search-result-date">' + matches[j].date + '</span></a>';
        }
        searchResults.innerHTML = html;
      }
      searchResults.style.display = 'block';
    });
  }

  var debounceTimer;
  searchInput.addEventListener('input', function() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function() {
      doSearch(searchInput.value.trim());
    }, 200);
  });

  document.addEventListener('click', function(e) {
    if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
      searchResults.style.display = 'none';
    }
  });

  searchInput.addEventListener('focus', function() {
    if (searchInput.value.trim().length >= 1) {
      doSearch(searchInput.value.trim());
    }
  });
})();

// Header shadow on scroll
(function() {
  var header = document.querySelector('header');
  if (!header) return;
  var ticking = false;
  window.addEventListener('scroll', function() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function() {
      if (window.scrollY > 10) {
        header.classList.add('scrolled');
      } else {
        header.classList.remove('scrolled');
      }
      ticking = false;
    });
  }, { passive: true });
})();

// Theme toggle
(function() {
  var toggle = document.getElementById('theme-toggle');
  if (!toggle) return;
  var iconMoon = toggle.querySelector('.icon-moon');
  var iconSun = toggle.querySelector('.icon-sun');
  var giscusThemeBase = toggle.getAttribute('data-giscus-theme-base') || '/css/giscus-';

  function updateIcons(theme) {
    if (theme === 'dark') {
      iconMoon.style.display = 'none';
      iconSun.style.display = 'block';
    } else {
      iconMoon.style.display = 'block';
      iconSun.style.display = 'none';
    }
  }

  updateIcons(document.documentElement.getAttribute('data-theme'));

  toggle.addEventListener('click', function() {
    var current = document.documentElement.getAttribute('data-theme');
    var next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateIcons(next);

    var giscusFrame = document.querySelector('iframe.giscus-frame');
    if (giscusFrame) {
      var giscusTheme = giscusThemeBase + (next === 'dark' ? 'dark' : 'light') + '.css';
      giscusFrame.contentWindow.postMessage(
        { giscus: { setConfig: { theme: giscusTheme } } },
        'https://giscus.app'
      );
    }
  });
})();
