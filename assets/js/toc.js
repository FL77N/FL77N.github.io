(function() {
  var toc = document.getElementById('toc');
  if (!toc) return;

  var allHeadings = document.querySelectorAll('article h1, article h2');
  var filteredHeadings = [];
  var skippedTitle = false;

  allHeadings.forEach(function(h) {
    if (!skippedTitle && h.tagName === 'H1') {
      skippedTitle = true;
      return;
    }
    filteredHeadings.push(h);
  });

  if (filteredHeadings.length === 0) {
    toc.style.display = 'none';
    return;
  }

  var hasH1 = false;
  filteredHeadings.forEach(function(h) {
    if (h.tagName === 'H1') hasH1 = true;
  });

  var idToLink = {};

  filteredHeadings.forEach(function(heading) {
    if (!heading.id) {
      heading.id = encodeURIComponent(
        heading.textContent.toLowerCase().replace(/[^a-z0-9一-鿿]/g, '-')
      );
    }

    var link = document.createElement('a');
    link.href = '#' + encodeURIComponent(heading.id);
    link.textContent = heading.textContent;
    link.setAttribute('data-heading-id', heading.id);

    idToLink[heading.id] = link;

    if (!hasH1) {
      link.classList.add('toc-link', 'level-1');

      var itemDiv = document.createElement('div');
      itemDiv.className = 'toc-parent';

      var toggleBtn = document.createElement('button');
      toggleBtn.className = 'toggle-btn';
      toggleBtn.style.visibility = 'hidden';

      itemDiv.appendChild(toggleBtn);
      itemDiv.appendChild(link);
      toc.appendChild(itemDiv);
    } else if (heading.tagName === 'H1') {
      link.classList.add('toc-link', 'level-1');

      var parentDiv = document.createElement('div');
      parentDiv.className = 'toc-parent';

      var toggleBtn = document.createElement('button');
      toggleBtn.className = 'toggle-btn';
      toggleBtn.textContent = '♠';
      toggleBtn.addEventListener('click', function() {
        var child = parentDiv.querySelector('.child-container');
        child.classList.toggle('open');
        toggleBtn.textContent = child.classList.contains('open') ? '♥' : '♠';
      });

      var childContainer = document.createElement('div');
      childContainer.className = 'child-container';

      parentDiv.appendChild(toggleBtn);
      parentDiv.appendChild(link);
      parentDiv.appendChild(childContainer);
      toc.appendChild(parentDiv);
    } else if (heading.tagName === 'H2') {
      link.classList.add('toc-link', 'level-2');

      var parentDiv = toc.lastChild;
      if (parentDiv && parentDiv.classList.contains('toc-parent') && parentDiv.querySelector('.child-container')) {
        parentDiv.querySelector('.child-container').appendChild(link);
      }
    }
  });

  // Scroll highlight (rAF-throttled)
  var tocTicking = false;
  window.addEventListener('scroll', function() {
    if (tocTicking) return;
    tocTicking = true;
    requestAnimationFrame(function() {
      var currentSection = '';
      var scrollY = window.scrollY + 100;

      filteredHeadings.forEach(function(section) {
        if (section.offsetTop <= scrollY) {
          currentSection = section.id;
        }
      });

      document.querySelectorAll('.toc-parent').forEach(function(parent) {
        var childContainer = parent.querySelector('.child-container');
        if (childContainer) {
          childContainer.classList.remove('open');
        }
        var btn = parent.querySelector('.toggle-btn');
        if (btn && btn.style.visibility !== 'hidden') btn.textContent = '♠';
      });

      document.querySelectorAll('.toc-link').forEach(function(link) {
        link.classList.remove('active');
      });

      var activeLink = idToLink[currentSection];
      if (activeLink) {
        activeLink.classList.add('active');
        var parent = activeLink.closest('.toc-parent');
        if (parent) {
          var child = parent.querySelector('.child-container');
          if (child) child.classList.add('open');
          var btn = parent.querySelector('.toggle-btn');
          if (btn && btn.style.visibility !== 'hidden') btn.textContent = '♥';
        }
      }
      tocTicking = false;
    });
  }, { passive: true });

  // Smooth scroll on click (offset for sticky header)
  document.querySelectorAll('.toc-link').forEach(function(link) {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      var targetId = link.getAttribute('data-heading-id');
      var targetElement = document.getElementById(targetId);
      if (targetElement) {
        var headerHeight = document.querySelector('header').offsetHeight + 20;
        var targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - headerHeight;
        window.scrollTo({ top: targetPosition, behavior: 'smooth' });
      }
    });
  });
})();

// FAB toggle (TOC drawer on mobile)
(function() {
  var fab = document.getElementById('toc-fab');
  var toc = document.getElementById('toc');
  if (!fab || !toc) return;

  // On mobile, relocate toc to be a direct child of <body> so its
  // position: fixed resolves against the viewport instead of any ancestor
  // that may have become a containing block (compositor layers, animations, etc.).
  var tocOriginalParent = toc.parentNode;
  var tocOriginalNext = toc.nextSibling;
  var mql = window.matchMedia('(max-width: 768px)');
  function syncTocLocation() {
    if (mql.matches) {
      if (toc.parentNode !== document.body) document.body.appendChild(toc);
    } else {
      if (toc.parentNode !== tocOriginalParent && tocOriginalParent) {
        tocOriginalParent.insertBefore(toc, tocOriginalNext);
      }
    }
  }
  syncTocLocation();
  if (mql.addEventListener) mql.addEventListener('change', syncTocLocation);
  else if (mql.addListener) mql.addListener(syncTocLocation);

  function pinDrawerToFab() {
    if (!mql.matches) return;
    var fabRect = fab.getBoundingClientRect();
    var gap = 8;
    var drawerHeight = toc.offsetHeight || 0;
    // viewport-relative top so drawer's bottom edge sits 8px above FAB top
    var topPx = fabRect.top - drawerHeight - gap;
    toc.style.setProperty('position', 'fixed', 'important');
    toc.style.setProperty('top', topPx + 'px', 'important');
    toc.style.setProperty('bottom', 'auto', 'important');
    toc.style.setProperty('left', '1rem', 'important');
  }

  fab.addEventListener('click', function() {
    toc.classList.toggle('toc-open');
    if (toc.classList.contains('toc-open')) {
      // measure after the .toc-open class applies its display:block
      requestAnimationFrame(pinDrawerToFab);
    }
  });
  window.addEventListener('resize', function() {
    if (toc.classList.contains('toc-open')) pinDrawerToFab();
  });
  document.addEventListener('click', function(e) {
    if (!fab.contains(e.target) && !toc.contains(e.target)) {
      toc.classList.remove('toc-open');
    }
  });
  toc.querySelectorAll('a').forEach(function(link) {
    link.addEventListener('click', function() {
      toc.classList.remove('toc-open');
    });
  });
})();
