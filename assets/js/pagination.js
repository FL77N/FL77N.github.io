(function() {
  function init(opts) {
    var items = opts.items;
    var perPage = opts.perPage;
    var paginationEl = opts.paginationEl;
    var onPageChange = opts.onPageChange || function() {};
    if (!items || !paginationEl) return null;

    var totalPages = Math.ceil(items.length / perPage);
    if (totalPages <= 1) return null;

    var currentPage = 1;

    function showPage(page) {
      currentPage = page;
      var start = (page - 1) * perPage;
      var end = start + perPage;
      for (var i = 0; i < items.length; i++) {
        items[i].style.display = (i >= start && i < end) ? '' : 'none';
      }
      render();
      onPageChange(page);
    }

    function render() {
      var html = '';
      if (currentPage > 1) {
        html += '<a class="nav-btn prev" href="javascript:void(0)">&lt;&lt; PREV</a>';
      }
      html += '<div class="page-control">';
      html += '<span class="page-label">PAGE</span>';
      html += '<input type="number" min="1" max="' + totalPages + '" value="' + currentPage + '" class="page-input">';
      html += '<span class="page-label">of ' + totalPages + '&nbsp;</span>';
      html += '<button class="nav-btn page-go">GO</button>';
      html += '</div>';
      if (currentPage < totalPages) {
        html += '<a class="nav-btn next" href="javascript:void(0)">NEXT &gt;&gt;</a>';
      }
      paginationEl.innerHTML = html;
      bind();
    }

    function bind() {
      var prev = paginationEl.querySelector('.prev');
      var next = paginationEl.querySelector('.next');
      var go = paginationEl.querySelector('.page-go');
      var input = paginationEl.querySelector('.page-input');

      if (prev) prev.addEventListener('click', function() { showPage(currentPage - 1); });
      if (next) next.addEventListener('click', function() { showPage(currentPage + 1); });
      if (go) go.addEventListener('click', function() {
        var v = parseInt(input.value);
        if (v >= 1 && v <= totalPages) showPage(v);
      });
      if (input) input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
          var v = parseInt(this.value);
          if (v >= 1 && v <= totalPages) showPage(v);
          return;
        }
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
          e.preventDefault();
          var v = parseInt(this.value);
          if (e.key === 'ArrowUp' && v > 1) this.value = v - 1;
          if (e.key === 'ArrowDown' && v < totalPages) this.value = v + 1;
        }
      });
    }

    showPage(1);
    return { goPage: showPage };
  }

  window.Pagination = { init: init };
})();
