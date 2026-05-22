(function() {
  if (!window.Pagination) return;
  window.__archivePagers = [];

  function refreshTagGroups(content) {
    content.querySelectorAll('.tag-group').forEach(function(tg) {
      var hasVisible = false;
      tg.querySelectorAll('li').forEach(function(li) {
        if (li.style.display !== 'none') hasVisible = true;
      });
      tg.style.display = hasVisible ? '' : 'none';
    });
  }

  document.querySelectorAll('.category-content').forEach(function(content) {
    var pager = window.Pagination.init({
      items: content.querySelectorAll('.post-list li'),
      perPage: 8,
      paginationEl: content.querySelector('.archive-pagination'),
      onPageChange: function() { refreshTagGroups(content); }
    });
    if (pager) window.__archivePagers.push(pager);
  });
})();

(function() {
  function expandFromHash() {
    var m = location.hash.match(/^#category-(.+)$/);
    if (!m) return;
    var input = document.getElementById('category-' + m[1]);
    if (!input) return;
    input.checked = true;
    setTimeout(function() {
      var group = input.closest('.category-group');
      if (group && group.scrollIntoView) {
        group.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }, 80);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', expandFromHash);
  } else {
    expandFromHash();
  }
  window.addEventListener('hashchange', expandFromHash);
})();
