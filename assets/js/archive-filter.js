(function() {
  var input = document.getElementById('archive-filter-input');
  var countEl = document.getElementById('archive-filter-count');
  var filterPagEl = document.getElementById('archive-filter-pagination');
  if (!input) return;

  var allLis = Array.prototype.slice.call(document.querySelectorAll('.archive-container li[data-search]'));
  var allTagGroups = Array.prototype.slice.call(document.querySelectorAll('.tag-group'));
  var allCategoryGroups = Array.prototype.slice.call(document.querySelectorAll('.category-group'));
  var allPaginations = Array.prototype.slice.call(document.querySelectorAll('.archive-pagination'));

  var FILTER_PER_PAGE = 10;
  var filterPager = null;
  var filterPagedItems = null;

  function refreshFilterGroups() {
    allTagGroups.forEach(function(g) {
      var anyVisible = false;
      g.querySelectorAll('li[data-search]').forEach(function(li) {
        if (li.style.display !== 'none') anyVisible = true;
      });
      g.style.display = anyVisible ? '' : 'none';
    });
    allCategoryGroups.forEach(function(cg) {
      var anyVisible = false;
      cg.querySelectorAll('li[data-search]').forEach(function(li) {
        if (li.style.display !== 'none') anyVisible = true;
      });
      cg.style.display = anyVisible ? '' : 'none';
      var toggle = cg.querySelector('.category-toggle');
      if (toggle) toggle.checked = anyVisible;
    });
  }

  function destroyFilterPager() {
    if (filterPagedItems) {
      filterPagedItems.forEach(function(li) { li.style.display = ''; });
      filterPagedItems = null;
    }
    if (filterPagEl) filterPagEl.innerHTML = '';
    filterPager = null;
  }

  function reset() {
    destroyFilterPager();
    allLis.forEach(function(li) { li.style.display = ''; });
    allTagGroups.forEach(function(g) { g.style.display = ''; });
    allCategoryGroups.forEach(function(cg) {
      cg.style.display = '';
      var toggle = cg.querySelector('.category-toggle');
      if (toggle) toggle.checked = false;
    });
    allPaginations.forEach(function(p) { p.style.display = ''; });
    if (window.__archivePagers) {
      window.__archivePagers.forEach(function(p) { if (p && p.goPage) p.goPage(1); });
    }
    countEl.textContent = '';
  }

  function applyFilterPagination(visibleLis) {
    destroyFilterPager();
    if (!filterPagEl || visibleLis.length <= FILTER_PER_PAGE) return;
    filterPagedItems = visibleLis;
    filterPager = window.Pagination.init({
      items: visibleLis,
      perPage: FILTER_PER_PAGE,
      paginationEl: filterPagEl,
      onPageChange: refreshFilterGroups
    });
  }

  function filter(q) {
    q = (q || '').trim().toLowerCase();
    if (!q) { reset(); return; }

    destroyFilterPager();
    allPaginations.forEach(function(p) { p.style.display = 'none'; });

    var visibleLis = [];
    allLis.forEach(function(li) {
      var hit = li.getAttribute('data-search').indexOf(q) !== -1;
      li.style.display = hit ? '' : 'none';
      if (hit) visibleLis.push(li);
    });

    refreshFilterGroups();

    var visible = visibleLis.length;
    countEl.textContent = '(' + visible + ' match' + (visible === 1 ? '' : 'es') + ')';

    applyFilterPagination(visibleLis);
  }

  input.addEventListener('input', function() { filter(input.value); });
})();
