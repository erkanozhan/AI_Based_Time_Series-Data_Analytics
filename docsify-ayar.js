// Web sürümünün (index.html) ve yazdırma sayfasının (yazdir.html) ortak docsify ayarları.
// yazdir.html bu dosyadan önce window.NOT_YAZDIR = true tanımlar.
(function () {
  var YAZDIR = !!window.NOT_YAZDIR;
  var DEPO = 'https://github.com/erkanozhan/AI_Based_Time_Series-Data_Analytics';

  // Bölüm dosyalarında en üst başlık "##" düzeyindedir (GitHub'daki görünüm için).
  // Sitede bölüm başlığı en büyük başlık olsun diye, belgede "#" başlık yoksa tüm başlıklar bir düzey yükseltilir.
  // Kod blokları (```...```) içindeki "#" ile başlayan yorum satırlarına dokunulmaz.
  function basliklariYukselt(md) {
    var satirlar = md.split('\n');
    var kodIcinde = false, h1Var = false;
    satirlar.forEach(function (s) {
      if (/^(\s*>)*\s*(```|~~~)/.test(s)) { kodIcinde = !kodIcinde; return; }
      if (!kodIcinde && /^# /.test(s)) h1Var = true;
    });
    if (h1Var) return md;
    kodIcinde = false;
    return satirlar.map(function (s) {
      if (/^(\s*>)*\s*(```|~~~)/.test(s)) { kodIcinde = !kodIcinde; return s; }
      return kodIcinde ? s : s.replace(/^#(#{1,5}\s)/, '$1');
    }).join('\n');
  }

  // Codes/ ve data/ klasörlerindeki dosyalara (.R, .py, .ipynb, .inp, .csv ...) verilen bağlantılar
  // sitede bir not sayfası gibi açılamaz; bunlar GitHub'daki dosyaya yönlendirilir.
  // dosya: bağlantının bulunduğu not dosyasının depo içindeki yolu (ör. "notlar/06_....md").
  function depoBaglantilari(md, dosya) {
    var klasor = (dosya || '').replace(/[^\/]*$/, '');
    return md.replace(/(!?)\[([^\]\n]*)\]\(([^)\s]+)\)/g, function (tum, resim, metin, hedef) {
      if (resim || /^(https?:|mailto:|#|\/)/.test(hedef)) return tum;
      var parcalar = [];
      (klasor + hedef).split('/').forEach(function (p) {
        if (p === '..') parcalar.pop(); else if (p && p !== '.') parcalar.push(p);
      });
      var yol = parcalar.join('/');
      if (!/^(Codes|data)(\/|#|$)/.test(yol)) return tum;
      return '[' + metin + '](' + DEPO + '/blob/main/' + yol + ')';
    });
  }

  // GitHub'ın satır içi formül yazımı $`...`$ biçimindedir. Formül eklentisi ters tırnakları formülün
  // parçası sanıp tırnak işareti olarak çizdiği için bunlar sitede düz $...$ biçimine çevrilir.
  function githubFormulleri(md) {
    return md.replace(/\$`([^`\n]+?)`\$/g, function (_, formul) { return '$' + formul + '$'; });
  }

  // Bir not dosyasına sitede uygulanan tüm dönüşümler
  function notuDuzenle(md, dosya) {
    return githubFormulleri(basliklariYukselt(depoBaglantilari(md, dosya)));
  }

  // Formül eklentisinden ÖNCE çalışması gerektiği için index.html bu eklentiyi, formül betiği
  // yüklendikten sonra listenin başına ekler. yazdir.html ise her bölüme notuDuzenle'yi kendisi uygular.
  function notDuzenleEklentisi(hook, vm) {
    hook.beforeEach(function (md) {
      return notuDuzenle(md, (vm.route && vm.route.file) || '');
    });
  }

  window.NOT_ARACLARI = { notuDuzenle: notuDuzenle, eklenti: notDuzenleEklentisi };

  // GitHub uyarı kutularını (> [!CAUTION] vb.) docsify'da da renkli göster
  function uyariKutulari(hook) {
    var basliklar = { NOTE: 'Not', TIP: 'İpucu', IMPORTANT: 'Önemli', WARNING: 'Uyarı', CAUTION: 'Dikkat' };
    hook.afterEach(function (html) {
      return html.replace(/<blockquote>\s*<p>\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(<br\s*\/?>)?/g, function (_, tur) {
        return '<blockquote class="uyari uyari-' + tur.toLowerCase() + '"><p><span class="uyari-baslik">' + basliklar[tur] + '</span>';
      });
    });
  }

  window.$docsify = {
    name: 'Zaman Serisi Analizi',
    repo: DEPO,
    homepage: 'Course_notes.md',
    loadSidebar: !YAZDIR,
    alias: { '/.*/_sidebar.md': '/_sidebar.md' },
    subMaxLevel: 2,
    relativePath: true,
    auto2top: true,
    executeScript: true,
    search: {
      paths: 'auto',
      depth: 3,
      placeholder: 'Notta ara',
      noData: 'Sonuç bulunamadı'
    },
    plugins: [uyariKutulari],
    markdown: {
      renderer: {
        code: function (code, lang) {
          // ```math blokları (GitHub'ın display formül biçimi) → KaTeX ile çiz
          if (lang === 'math' && window.katex) {
            return '<div class="math-blok">' +
              window.katex.renderToString(code, { displayMode: true, throwOnError: false }) +
              '</div>';
          }
          return this.origin.code.apply(this, arguments);
        }
      }
    }
  };
})();
