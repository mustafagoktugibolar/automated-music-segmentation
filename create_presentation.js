"use strict";

const pptxgen = require("pptxgenjs");
const pres = new pptxgen();

pres.layout = "LAYOUT_WIDE";
pres.title = "Automated Music Structure Segmentation";
pres.author = "Capstone Team";

// ─── COLOR PALETTE ──────────────────────────────────────────────────────────
const C = {
  navy:     "1B2A4A",
  navyMid:  "243556",
  copper:   "B87333",
  copperLt: "D4944A",
  slate:    "4A5568",
  white:    "FFFFFF",
  offWhite: "F4F6FA",
  gray:     "718096",
  grayLt:   "E2E8F0",
  green:    "2D6A4F",
  greenLt:  "52B788",
  amber:    "B45309",
  amberLt:  "F59E0B",
  teal:     "0D9488",
  red:      "9B1C1C",
  purple:   "6D28D9",
};

// ─── PRESENTER ASSIGNMENTS ───────────────────────────────────────────────────
// Slide 1–7 → Presenter 1, 8–14 → Presenter 2, 15–20 → Presenter 3
const PRESENTER_MAP = {
  1:"Presenter 1",2:"Presenter 1",3:"Presenter 1",4:"Presenter 1",
  5:"Presenter 1",6:"Presenter 1",7:"Presenter 1",
  8:"Presenter 2",9:"Presenter 2",10:"Presenter 2",11:"Presenter 2",
  12:"Presenter 2",13:"Presenter 2",14:"Presenter 2",
  15:"Presenter 3",16:"Presenter 3",17:"Presenter 3",18:"Presenter 3",
  19:"Presenter 3",20:"Presenter 3",
};
const PRESENTER_COLORS = {
  "Presenter 1": C.copper,
  "Presenter 2": C.teal,
  "Presenter 3": C.green,
};

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function badge(slide, n) {
  const who = PRESENTER_MAP[n];
  const col = PRESENTER_COLORS[who];
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:10.5, y:7.0, w:2.6, h:0.34,
    fill:{color:col}, line:{color:col}, rounding:0.08,
  });
  slide.addText(`🎤 ${who}`, {
    x:10.5, y:7.0, w:2.6, h:0.34,
    fontSize:9, color:C.white, bold:true,
    align:"center", valign:"middle", margin:0,
  });
  slide.addText(`${n} / 20`, {
    x:0.2, y:7.05, w:1, h:0.22,
    fontSize:8, color:C.gray, align:"left", margin:0,
  });
}

function header(slide, title, subtitle) {
  slide.background = {color:C.offWhite};
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0, y:0, w:13.3, h:1.15,
    fill:{color:C.navy}, line:{color:C.navy},
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0, y:1.15, w:13.3, h:0.055,
    fill:{color:C.copper}, line:{color:C.copper},
  });
  slide.addText(title, {
    x:0.4, y:0.08, w:11.5, h:0.66,
    fontSize:26, color:C.white, bold:true, fontFace:"Georgia",
    align:"left", valign:"middle", margin:0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x:0.4, y:0.73, w:11.5, h:0.38,
      fontSize:13, color:C.copperLt, fontFace:"Calibri",
      align:"left", valign:"middle", margin:0,
    });
  }
}

function divider(slide, title, sub) {
  slide.background = {color:C.navy};
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0, y:2.9, w:13.3, h:0.06,
    fill:{color:C.copper}, line:{color:C.copper},
  });
  slide.addText(title, {
    x:1, y:1.7, w:11.3, h:1.2,
    fontSize:40, color:C.white, bold:true, fontFace:"Georgia",
    align:"center", valign:"middle",
  });
  if (sub) {
    slide.addText(sub, {
      x:1.5, y:3.2, w:10.3, h:0.9,
      fontSize:16, color:C.copperLt, fontFace:"Calibri",
      align:"center",
    });
  }
}

// Reasoning card: Problem → Decision → Why it works
function reasonCard(slide, x, y, w, h, problem, decision, why, accentColor) {
  const ac = accentColor || C.copper;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill:{color:C.white}, line:{color:C.grayLt},
    shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.08},
    rounding:0.08,
  });
  slide.addShape(pres.shapes.RECTANGLE, {x, y, w, h:0.055, fill:{color:ac}, line:{color:ac}});

  const rh = (h - 0.1) / 3;
  const labels = ["Problem", "Decision", "Why it works"];
  const vals   = [problem, decision, why];
  const cols   = [C.red, C.navy, C.green];
  labels.forEach((lbl, i) => {
    const iy = y + 0.12 + i * rh;
    slide.addText(lbl.toUpperCase(), {
      x:x+0.14, y:iy, w:w-0.28, h:0.22,
      fontSize:8, color:cols[i], bold:true, fontFace:"Calibri",
      align:"left", valign:"middle", margin:0,
    });
    slide.addText(vals[i], {
      x:x+0.14, y:iy+0.22, w:w-0.28, h:rh-0.28,
      fontSize:11, color:C.slate, fontFace:"Calibri",
      align:"left", valign:"top", margin:0,
    });
    if (i < 2) {
      slide.addShape(pres.shapes.RECTANGLE, {
        x:x+0.14, y:iy+rh-0.04, w:w-0.28, h:0.01,
        fill:{color:C.grayLt}, line:{color:C.grayLt},
      });
    }
  });
}

// Simple bullet text box
function bullets(slide, items, x, y, w, h, fontSize, color) {
  const rich = items.map((txt, i) => ({
    text: txt,
    options: {
      bullet:true,
      breakLine: i < items.length - 1,
      fontSize: fontSize || 13,
      color: color || C.slate,
      fontFace:"Calibri",
    },
  }));
  slide.addText(rich, {x, y, w, h, valign:"top", margin:[6,10,6,10]});
}

// Arrow flow: array of {label, color} in a row
function flowRow(slide, items, x, y, itemW, itemH) {
  items.forEach(({label, color, textColor}, i) => {
    const ix = x + i * (itemW + 0.22);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x:ix, y, w:itemW, h:itemH,
      fill:{color:color||C.navy}, line:{color:color||C.navy}, rounding:0.09,
    });
    slide.addText(label, {
      x:ix, y, w:itemW, h:itemH,
      fontSize:10, color:textColor||C.white, bold:true, fontFace:"Calibri",
      align:"center", valign:"middle",
    });
    if (i < items.length - 1) {
      slide.addText("→", {
        x:ix+itemW, y:y+itemH/2-0.15, w:0.22, h:0.3,
        fontSize:13, color:C.copper, bold:true, align:"center",
      });
    }
  });
}

// Highlight quote bar
function quoteBar(slide, text, y) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x:0, y, w:13.3, h:0.46,
    fill:{color:C.navy}, line:{color:C.navy},
  });
  slide.addText(text, {
    x:0.4, y, w:12.5, h:0.46,
    fontSize:11, color:C.copperLt, fontFace:"Calibri",
    align:"center", valign:"middle",
  });
}

// Comparison: two-column YES vs NO / A vs B
function vsBox(slide, leftTitle, leftItems, rightTitle, rightItems, x, y, w, h, leftColor, rightColor) {
  const colW = w / 2 - 0.05;
  const lc = leftColor || C.green;
  const rc = rightColor || C.red;
  // left
  slide.addShape(pres.shapes.RECTANGLE, {x, y, w:colW, h, fill:{color:C.white}, line:{color:C.grayLt}});
  slide.addShape(pres.shapes.RECTANGLE, {x, y, w:colW, h:0.45, fill:{color:lc}, line:{color:lc}});
  slide.addText(leftTitle, {x, y, w:colW, h:0.45, fontSize:13, color:C.white, bold:true, fontFace:"Georgia", align:"center", valign:"middle"});
  bullets(slide, leftItems, x+0.1, y+0.52, colW-0.2, h-0.6, 11, C.slate);
  // right
  const rx = x + colW + 0.1;
  slide.addShape(pres.shapes.RECTANGLE, {x:rx, y, w:colW, h, fill:{color:C.white}, line:{color:C.grayLt}});
  slide.addShape(pres.shapes.RECTANGLE, {x:rx, y, w:colW, h:0.45, fill:{color:rc}, line:{color:rc}});
  slide.addText(rightTitle, {x:rx, y, w:colW, h:0.45, fontSize:13, color:C.white, bold:true, fontFace:"Georgia", align:"center", valign:"middle"});
  bullets(slide, rightItems, rx+0.1, y+0.52, colW-0.2, h-0.6, 11, C.slate);
}


// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = {color:C.navy};
  s.addShape(pres.shapes.OVAL, {x:10.8,y:-0.6,w:4,h:4, fill:{color:"1F3460"}, line:{color:"1F3460"}});
  s.addShape(pres.shapes.OVAL, {x:-1.2,y:5.5,w:3,h:3, fill:{color:"162440"}, line:{color:"162440"}});
  s.addShape(pres.shapes.RECTANGLE, {x:0,y:4.6,w:13.3,h:0.07, fill:{color:C.copper}, line:{color:C.copper}});

  s.addText("Automated Music", {x:0.7,y:0.9,w:11,h:1.0, fontSize:46,color:C.white,bold:true,fontFace:"Georgia",align:"center"});
  s.addText("Structure Segmentation", {x:0.7,y:1.85,w:11,h:1.0, fontSize:46,color:C.copperLt,bold:true,fontFace:"Georgia",align:"center"});
  s.addText("Neden çoklu algoritma, neden iki seviye fusion, neden dağıtık mimari?", {
    x:1,y:3.0,w:11.3,h:0.55, fontSize:14,color:C.grayLt,fontFace:"Calibri",align:"center",
  });
  s.addText("Capstone Final Presentation  •  Spring 2026", {
    x:1,y:3.62,w:11.3,h:0.4, fontSize:12,color:C.gray,fontFace:"Calibri",align:"center",
  });

  const ps = ["Presenter 1","Presenter 2","Presenter 3"];
  const pc = [C.copper, C.teal, C.green];
  ps.forEach((p,i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x:2.4+i*3.2, y:4.9, w:2.8, h:0.4,
      fill:{color:pc[i]}, line:{color:pc[i]}, rounding:0.1,
    });
    s.addText(p, {x:2.4+i*3.2,y:4.9,w:2.8,h:0.4, fontSize:12,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
  });
  badge(s,1);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — SUNUM HİKAYESİ (neden bu sırayla anlatıyoruz?)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Sunumun Mantığı", "Her bölüm bir öncekinin sorusunu cevaplayacak");
  badge(s,2);

  const chain = [
    {num:"01", q:"Problem nedir?", a:"Müzik yapısındaki geçiş noktaları nasıl bulunur?", color:C.copper},
    {num:"02", q:"Neden zor?", a:"Tek bir sinyal her geçişi göremez — çok boyutlu kanıt gerekli", color:C.teal},
    {num:"03", q:"Nasıl bir sistem?", a:"Dağıtık, asenkron, her algoritma bağımsız çalışır", color:C.green},
    {num:"04", q:"Tek algoritma yeterli değil mi?", a:"Custom pipeline içinde 6 sinyal birleşiyor: feature-level fusion", color:C.amber},
    {num:"05", q:"Bir pipeline hâlâ yetmez mi?", a:"Farklı algoritmaların hatalarını dengelemek için algorithm-level fusion", color:C.purple},
    {num:"06", q:"Sonuç doğru mu nasıl biliriz?", a:"SALAMI anotasyonlarıyla iki farklı toleransta ölçüm", color:C.navy},
  ];

  chain.forEach(({num,q,a,color},i) => {
    const col = i % 2;
    const row = Math.floor(i/2);
    const x = 0.3 + col*6.55;
    const y = 1.45 + row*1.9;

    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:1.7, fill:{color:C.white}, line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:4,offset:1,angle:135,opacity:0.07}});
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:0.07,h:1.7, fill:{color}, line:{color}});
    s.addText(num, {x:x+0.2,y:y+0.08,w:0.6,h:0.6, fontSize:22,color,bold:true,fontFace:"Georgia",align:"center",margin:0});
    s.addText("SORU: "+q, {x:x+0.85,y:y+0.1,w:5.1,h:0.42, fontSize:12,color:C.navy,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText("CEVAP: "+a, {x:x+0.85,y:y+0.58,w:5.1,h:0.95, fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — PROBLEM: SINIR TESPİTİ NE DEMEKTİR?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Problem: Boundary Detection", "Müzikal yapı değişimlerinin tam zamanını bulmak");
  badge(s,3);

  // Timeline
  const segs = [
    {label:"Intro",color:C.copper,w:2.0},
    {label:"Verse",color:C.teal,w:2.7},
    {label:"Chorus",color:C.green,w:2.4},
    {label:"Verse",color:C.teal,w:2.4},
    {label:"Chorus",color:C.green,w:2.4},
    {label:"Outro",color:C.amber,w:1.3},
  ];
  let cx = 0.3;
  segs.forEach(({label,color,w}) => {
    s.addShape(pres.shapes.RECTANGLE, {x:cx,y:1.45,w,h:0.85, fill:{color}, line:{color}});
    s.addText(label, {x:cx,y:1.45,w,h:0.85, fontSize:12,color:C.white,bold:true,align:"center",valign:"middle"});
    cx += w;
  });
  // boundary arrows
  const bx = [2.3, 5.0, 7.4, 9.8];
  bx.forEach(x => {
    s.addShape(pres.shapes.RECTANGLE, {x:x+0.3,y:1.2,w:0.04,h:1.35, fill:{color:"DC2626"}, line:{color:"DC2626"}});
    s.addText("◀ boundary", {x:x,y:2.6,w:1.0,h:0.25, fontSize:8,color:"DC2626",align:"center"});
  });

  // Three key clarifications
  const points = [
    {icon:"🎯",title:"Primary task: boundary, NOT label",
     body:"Bilgisayar için ilk adım section ismi koymak değil, geçişin gerçekleştiği saniyeyi bulmaktır. Label bu karardan sonra gelir.",color:C.copper},
    {icon:"📏",title:"Segment = iki boundary arasındaki aralık",
     body:"Bir boundary bulununca komşu boundary ile birleşerek segment oluşur. Segment sayısı boundary sayısına bağlıdır.",color:C.teal},
    {icon:"⚖️",title:"Fazla boundary da az boundary da hata",
     body:"Over-segmentation: gereksiz fazla kesim → precision düşer. Under-segmentation: geçiş kaçırılır → recall düşer.",color:C.green},
  ];
  points.forEach(({icon,title,body,color},i) => {
    const x = 0.3 + i*4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x,y:3.05,w:4.05,h:3.55,
      fill:{color:C.white}, line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.07}, rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE, {x,y:3.05,w:4.05,h:0.06, fill:{color}, line:{color}});
    s.addText(icon, {x:x+0.15,y:3.15,w:0.55,h:0.55, fontSize:24,align:"center",valign:"middle"});
    s.addText(title, {x:x+0.75,y:3.15,w:3.15,h:0.55, fontSize:12,color,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText(body, {x:x+0.15,y:3.78,w:3.75,h:2.7, fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — NEDEN ZOR? (tek sinyal neden yetmez)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Neden Zor? — Tek Sinyal Yetmez", "Her müzikal geçiş farklı bir akustik kanıt bırakır");
  badge(s,4);

  // Central insight
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:1.42,w:12.7,h:0.75,
    fill:{color:"EFF6FF"}, line:{color:C.teal}, rounding:0.07,
  });
  s.addText("Temel çelişki: Verse → Chorus geçişinde bazen enerji yükselir, bazen akort değişir, bazen sadece davul girer — hiçbiri her zaman olmaz.", {
    x:0.5,y:1.42,w:12.3,h:0.75, fontSize:13,color:C.navy,fontFace:"Calibri",align:"center",valign:"middle",
  });

  const examples = [
    {signal:"Sadece RMS (enerji)",problem:"Verse'ten Chorus'a geçişte enerji artabilir ama acoustic guitar'lı iki verse arasında artmaz",verdict:"YETERSİZ",color:C.red},
    {signal:"Sadece Chroma (armoni)",problem:"Aynı chord progression'a sahip iki farklı section harmonik olarak aynı görünür",verdict:"YETERSİZ",color:C.red},
    {signal:"Sadece Onset (vuruş)",problem:"Drum pattern içinde yüzlerce onset var; section boundary'den çok nota başlangıçlarını gösterir",verdict:"YETERSİZ",color:C.red},
    {signal:"SSM (öz-benzerlik matrisi)",problem:"Tekrar eden Chorus'ları yakalayabilir ama timestamp kesinliği frame çözünürlüğüne bağlıdır",verdict:"GÜÇLÜ ama TEK BAŞINA EKSİK",color:C.amber},
  ];

  examples.forEach(({signal,problem,verdict,color},i) => {
    const col = i % 2;
    const row = Math.floor(i/2);
    const x = 0.3 + col*6.55;
    const y = 2.38 + row*2.2;
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:2.0, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:0.45, fill:{color:C.navyMid}, line:{color:C.navyMid}});
    s.addText("📊 "+signal, {x:x+0.1,y,w:5.9,h:0.45, fontSize:12,color:C.white,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText(problem, {x:x+0.1,y:y+0.5,w:5.9,h:1.0, fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:x+0.1,y:y+1.58,w:5.9,h:0.32, fill:{color}, line:{color}, rounding:0.05});
    s.addText("→ "+verdict, {x:x+0.1,y:y+1.58,w:5.9,h:0.32, fontSize:10,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
  });

  quoteBar(s,"Sonuç: birden fazla sinyali birleştirmezsek, her genre'da güvenilir bir sistem kuramayız", 6.72);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — NEDEN DAĞITIK MİMARİ? (design decision)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Neden Dağıtık Mimari Seçtik?", "DSP hesaplama HTTP isteğini bloke etmemeli");
  badge(s,5);

  vsBox(s,
    "❌ Monolitik Yaklaşım (API içinde hesapla)",
    [
      "HTTP request, 4 algoritmanın bitmesini bekler",
      "Bir algoritma yavaşlarsa tüm kullanıcı bloke olur",
      "Algoritma eklemek API'yi doğrudan değiştirir",
      "Paralel çalışma zor — thread blocking riski",
      "Scale etmek için tüm servisi büyütmek gerekir",
    ],
    "✓ Dağıtık Worker Yaklaşımı (seçtiğimiz)",
    [
      "API sadece task kaydeder ve mesaj yayınlar",
      "Algoritmalar birbirinden bağımsız, paralel çalışır",
      "Yeni algoritma = yeni worker, API değişmez",
      "Her worker ayrı scale edilebilir",
      "Bir worker crash olsa diğerleri sonuç üretir",
    ],
    0.3, 1.42, 12.7, 4.2, C.red, C.green
  );

  // Why RabbitMQ specifically
  s.addShape(pres.shapes.RECTANGLE, {x:0.3,y:5.82,w:12.7,h:0.45, fill:{color:C.navyMid}, line:{color:C.navyMid}});
  s.addText("RabbitMQ'yu Neden Seçtik?", {x:0.3,y:5.82,w:12.7,h:0.45, fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});

  const mq = [
    ["Topic Exchange", "Her algoritma kendi routing key'ine bağlı — tek mesaj birden fazla worker'a gitmez"],
    ["ACK mekanizması", "Worker mesajı işleyene kadar queue'da tutar; crash olursa tekrar teslim edilir"],
    ["Servis izolasyonu", "Backend ve worker'lar birbirini doğrudan çağırmaz — birinin durması diğerini etkilemez"],
    ["Tek sonuç kanalı", "Tüm worker'lar segmentation.result'a yazar — ResultListener tek noktada toplar"],
  ];
  mq.forEach(([title,desc],i) => {
    const col = i%2, row = Math.floor(i/2);
    const x = 0.3+col*6.55, y = 6.38+row*0.58;
    s.addText("▸ "+title+":", {x,y,w:2.2,h:0.5, fontSize:10,color:C.copperLt,bold:true,fontFace:"Calibri",align:"left",valign:"middle"});
    s.addText(desc, {x:x+2.2,y,w:4.2,h:0.5, fontSize:10,color:C.grayLt,fontFace:"Calibri",align:"left",valign:"middle"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — BİR REQUEST NASIL İLERLER? (sırayı neden böyle kurguladık)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Bir İstek Sistemde Nasıl İlerler?", "Her adımın sırası bilinçli bir tasarım kararıdır");
  badge(s,6);

  const steps = [
    {n:"1",title:"Validation önce gelir",reason:"Hatalı parametre worker'a kadar gidip geç hata vermemeli — API katmanında Pydantic şemalarıyla engellenir.",color:C.copper},
    {n:"2",title:"Task önce DB'ye yazılır, sonra publish edilir",reason:"Worker çok hızlı yanıt verse bile ResultListener'ın bulabileceği bir kayıt olsun. Yayın öncesi kayıt garantisi.",color:C.teal},
    {n:"3",title:"Fusion hemen dispatch edilmez",reason:"Fusion worker'ının işleyebilmesi için önce 4 base algoritma sonucu gerekir. Orchestrator fusion'ı 'expected' listesine ekler ama 'dispatch' etmez.",color:C.green},
    {n:"4",title:"Worker'lar paralel çalışır",reason:"RabbitMQ her algoritmaya ayrı routing key ile mesaj iletir. custom_librosa, Foote, CNMF, SCluster birbirini beklemez.",color:C.amber},
    {n:"5",title:"Sonuçlar normalize edilir — anında değil, gelince",reason:"ResultListener her sonucu alınca normalize eder ve DB'ye yazar. Frontend partial sonuçları SSE ile anlık görebilir.",color:C.purple},
    {n:"6",title:"Fusion ancak tüm base'ler tamamlanınca başlar",reason:"_maybe_dispatch_fusion() 4 baseline'ın tamamını bekler. Bu, yarım bilgiyle fusion yapılmasını önler.",color:C.navy},
  ];

  steps.forEach(({n,title,reason,color},i) => {
    const col = i%2, row = Math.floor(i/2);
    const x = 0.3+col*6.55, y = 1.45+row*1.92;
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:1.75, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:x+0.08,y:y+0.1,w:0.48,h:0.48, fill:{color}, line:{color}, rounding:0.08});
    s.addText(n, {x:x+0.08,y:y+0.1,w:0.48,h:0.48, fontSize:14,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
    s.addText(title, {x:x+0.65,y:y+0.1,w:5.3,h:0.42, fontSize:12,color:C.navy,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(reason, {x:x+0.1,y:y+0.62,w:5.9,h:1.05, fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — NEDEN ORTAK ŞEMA? (common result schema)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Neden Ortak Sonuç Şeması?", "Farklı algoritmalar aynı dili konuşmalı ki fusion mümkün olsun");
  badge(s,7);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:1.42,w:12.7,h:0.8,
    fill:{color:"FFF7ED"}, line:{color:C.amber}, rounding:0.07,
  });
  s.addText("Şema olmadan ne olurdu?  Fusion servisi her algoritma için ayrı bir parser yazmak zorunda kalırdı. custom_librosa'nın çıktısı, Foote'un çıktısı, CNMF'in çıktısı — hepsi farklı format. Yeni algoritma eklemek fusion'ı kırar.", {
    x:0.5,y:1.42,w:12.3,h:0.8, fontSize:12,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle",
  });

  reasonCard(s, 0.3, 2.42, 4.0, 4.95,
    "Her algoritma farklı raw format üretiyor (Foote: sadece boundary zamanları, CNMF: label'lı segment listesi, custom: feature metadata ile)",
    "AlgorithmResult şeması: task_id, status, worker_type, algorithm, duration, boundaries[], segments[], diagnostics{}",
    "Fusion ve evaluation hiçbir zaman algoritma-specific kod çalıştırmaz. Yeni bir worker yazmak şemaya uymaktan ibaretti.",
    C.copper
  );

  reasonCard(s, 4.5, 2.42, 4.0, 4.95,
    "Structural label 'A' mı Verse mı? MSAF'ın raw label'ı 'Verse' diye isimlendiriyordu. Bu iddia çok güçlü.",
    "İki katman ayrıldı: structural_label (A/B/C — benzerlik iddiası) ve semantic_label (Intro/Verse/Chorus — müzikal rol iddiası)",
    "Sistem bilmediği şeyi biliyormuş gibi göstermiyor. Her label'a confidence ve reason da eklendi.",
    C.teal
  );

  reasonCard(s, 8.7, 2.42, 4.3, 4.95,
    "ResultListener her sonucu aldığında hem frontend'e göstermek hem de fusion'a vermek zorunda — iki farklı ihtiyaç",
    "results[algo] → segment listesi (frontend için). results[algo__result] → full normalized result (fusion için). results[algo__diagnostics] → açıklanabilirlik için.",
    "Eski frontend bozulmadan yeni fusion sistemi çalışabildi. Geriye dönük uyumluluk ve genişletilebilirlik aynı anda sağlandı.",
    C.green
  );
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — SECTION DIVIDER
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  divider(s, "Segmentasyon Algoritmaları", "Neden bu 4 algoritmayı seçtik ve neden birlikte çalışıyorlar?");
  badge(s,8);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — CUSTOM LİBROSA: NEDEN MULTI-FEATURE?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Custom Librosa Pipeline — Neden Multi-Feature?", "Her sinyal farklı bir geçiş tipini görür");
  badge(s,9);

  s.addText("Pipeline'ı bu sırayla kurguladık:", {
    x:0.3,y:1.42,w:5,h:0.35, fontSize:12,color:C.navy,bold:true,fontFace:"Georgia",align:"left",
  });

  flowRow(s, [
    {label:"Ses Yükle\n(ffmpeg\nmono 22050Hz)",color:C.slate},
    {label:"Aktif Bölge\nBul\n(sessizliği kırp)",color:C.teal},
    {label:"Özellik\nÇıkar\n(Chroma+MFCC)",color:C.green},
    {label:"SSM Kur\n(her frame'i\nherkesle karşılaştır)",color:C.copper},
    {label:"Aday\nÜret\n(6 sinyal)",color:C.amber},
    {label:"Birleştir\n(feature fusion\n+ snapping)",color:C.purple},
  ], 0.3, 1.9, 2.0, 1.6);

  // Why each step in this order
  const whys = [
    {step:"1. Önce aktif bölge neden?",
     why:"Sessizlik SSM hesaplarını ve segment süresi ölçümlerini bozar. Analizi müzikal içerikle sınırlandırıp son adımda zaman eksenini geri düzeltiyoruz.",color:C.teal},
    {step:"2. SSM neden en ağır adım?",
     why:"Her frame her frame ile karşılaştırılır (O(N²) işlem). Tekrarlanan Chorus'lar matriste parlak köşegen bloklar olarak görünür — bu yapısal bilgiyi başka hiçbir sinyal sağlayamaz.",color:C.copper},
    {step:"3. 6 sinyal neden aynı anda?",
     why:"RMS enerji değişimini, Chroma armonik değişimi, Onset vuruş yoğunluğunu, Beat ritmik hizalamayı, SSM yapısal tekrarı, Lyrics metin sınırlarını yakapar. Biri diğerini göremez.",color:C.green},
    {step:"4. Snapping neden en son?",
     why:"SSM doğru bölgeyi bulur ama timestamp biraz kayık olabilir. En son adımda güçlü bir onset veya beat'e yaslarız — yapısal detector 'nerede', onset 'tam ne zaman' sorusunu cevaplar.",color:C.amber},
  ];

  whys.forEach(({step,why,color},i) => {
    const col = i%2, row = Math.floor(i/2);
    const x = 0.3+col*6.55, y = 3.75+row*1.65;
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:1.5, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:0.06,h:1.5, fill:{color}, line:{color}});
    s.addText(step, {x:x+0.18,y:y+0.08,w:5.8,h:0.38, fontSize:12,color,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(why, {x:x+0.18,y:y+0.5,w:5.8,h:0.95, fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — SSM NEDEN MERKEZİ SİNYAL?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Self-Similarity Matrix — Neden Merkezi Sinyal?", "Tekrar bilgisi yalnızca SSM'de var");
  badge(s,10);

  // SSM concept visual (simplified grid)
  s.addText("SSM'de ne görürüz?", {x:0.3,y:1.42,w:4,h:0.35, fontSize:13,color:C.navy,bold:true,fontFace:"Georgia"});

  // Draw a simplified SSM grid
  const gridSize = 8;
  const cellSize = 0.46;
  const gx = 0.3, gy = 1.88;
  // Simulate a song: Intro(0-1), Verse(1-3), Chorus(3-5), Verse(5-7), Chorus(7-8)
  const sectionOf = (i) => {
    if (i < 1) return 0; // Intro
    if (i < 3) return 1; // Verse
    if (i < 5) return 2; // Chorus
    if (i < 7) return 1; // Verse (same as 1)
    return 2; // Chorus (same as 2)
  };
  const brightness = (r,c) => sectionOf(r) === sectionOf(c) ? "2D6A4F" : (r===c ? "1B2A4A" : "E2E8F0");
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      s.addShape(pres.shapes.RECTANGLE, {
        x:gx+c*cellSize, y:gy+r*cellSize, w:cellSize-0.02, h:cellSize-0.02,
        fill:{color:brightness(r,c)}, line:{color:"FFFFFF"},
      });
    }
  }
  const labels = ["Intro","Verse","Verse","Chorus","Chorus","Verse","Verse","Chorus"];
  labels.forEach((l,i) => {
    s.addText(l, {x:gx+i*cellSize+0.01,y:gy+gridSize*cellSize+0.04,w:cellSize,h:0.3, fontSize:6.5,color:C.slate,align:"center"});
    s.addText(l, {x:gx-0.55,y:gy+i*cellSize,w:0.54,h:cellSize, fontSize:6.5,color:C.slate,align:"right",valign:"middle"});
  });
  s.addText("🟩 = benzer bölgeler (Verse-Verse, Chorus-Chorus)", {
    x:gx,y:gy+gridSize*cellSize+0.42,w:gridSize*cellSize,h:0.35, fontSize:9,color:C.green,fontFace:"Calibri",align:"center",
  });

  // Right: Why SSM
  const ssmPoints = [
    {title:"Anlık değişim değil, yapısal tekrar",
     body:"RMS veya onset anlık olayları ölçer. SSM ise 'bu parçanın farklı iki anı birbirine benziyor mu?' sorusunu sorar. Chorus'un 3. kez geldiğini yalnızca SSM fark edebilir.",color:C.copper},
    {title:"Neden en yüksek ağırlık (0.42)?",
     body:"Feature fusion'da SSM candidate'ı tek başına threshold'u geçebilir (confidence ≥ 0.5 yeterliyse). Diğer sinyaller destek sağlar; SSM olmadan gerçek section boundary'yi kaçırma riski çok yüksek.",color:C.green},
    {title:"Transposition invariance neden var?",
     body:"Aynı motif farklı tonda tekrar edebilir. 12 pitch shift deneyip en yüksek benzerliği alırız. Böylece key change olsa bile yapısal tekrar kaybolmaz.",color:C.teal},
    {title:"Neden smoothing ve diagonal enhancement?",
     body:"Ham SSM gürültülüdür. Diagonal boyunca smooth edip tempo değişkenliğini hesaba katarak tekrar yollarını güçlendiririz. Checkerboard kernel boundary'de en yüksek tepkiyi verir.",color:C.amber},
  ];

  ssmPoints.forEach(({title,body,color},i) => {
    const y = 1.42 + i*1.55;
    s.addShape(pres.shapes.RECTANGLE, {x:4.4,y,w:8.7,h:1.42, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE, {x:4.4,y,w:0.06,h:1.42, fill:{color}, line:{color}});
    s.addText(title, {x:4.58,y:y+0.08,w:8.4,h:0.38, fontSize:12,color,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(body, {x:4.58,y:y+0.5,w:8.4,h:0.88, fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — MSAF BASELINE'LAR: NEDEN 3 FARKLI YÖNTEM?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "MSAF Baseline'ları — Neden 3 Farklı Yöntem?", "Farklı matematiksel varsayımlar, farklı hata profilleri");
  badge(s,11);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"EFF6FF"}, line:{color:C.teal}, rounding:0.07,
  });
  s.addText("Fusion'ın değeri: aynı algoritmanın 4 parametreli versiyonunu birleştirmek değil, farklı matematiksel perspektiften gelen bağımsız görüşleri birleştirmek. Benzer hata profilleri fusion'ı anlamsız kılar.", {
    x:0.5,y:1.42,w:12.3,h:0.72, fontSize:12,color:C.navy,fontFace:"Calibri",align:"left",valign:"middle",
  });

  const algos = [
    {
      name:"Foote",
      math:"Local Novelty Detection",
      perspective:"'Bu anda öncesine göre ne kadar değişim var?'",
      strength:"Keskin, yerel geçişleri iyi yakalar. Hızlı ve açıklanabilir.",
      weakness:"Davul fill gibi yapısal olmayan geçici ani değişimlere duyarlı.",
      weight:"0.15 — yalnızca yerel sinyal, yapısal bilgi eksik",
      color:"5B21B6",
    },
    {
      name:"CNMF",
      math:"Convex Non-negative Matrix Factorization",
      perspective:"'Parçada gizli tekrar eden bileşenler hangileri?'",
      strength:"Latent pattern'ları factorize eder — verinin zaten içindeki yapıyı çıkarır.",
      weakness:"Factorization rank (k) seçimi sınır sayısını etkiler.",
      weight:"0.20 — Foote'tan daha az yerel gürültüye duyarlı",
      color:C.teal,
    },
    {
      name:"SCluster",
      math:"Spectral Clustering on Affinity Graph",
      perspective:"'Tüm parçada global gruplar nasıl bölünüyor?'",
      strength:"Yerel değil, global benzerlik yapısını kullanır. Tekrar eden section'ları cluster olarak yakalar.",
      weakness:"Cluster sayısı parametresi segment granülaritesini belirler.",
      weight:"0.30 — global yapı, custom'dan sonra ikinci en yüksek",
      color:C.green,
    },
  ];

  algos.forEach(({name,math,perspective,strength,weakness,weight,color},i) => {
    const x = 0.3+i*4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x,y:2.35,w:4.05,h:5.25,
      fill:{color:C.white}, line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.07}, rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE, {x,y:2.35,w:4.05,h:0.55, fill:{color}, line:{color}});
    s.addText(name, {x,y:2.35,w:4.05,h:0.55, fontSize:20,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle",margin:0});
    s.addText(math, {x:x+0.12,y:2.97,w:3.81,h:0.38, fontSize:10,color,bold:true,fontFace:"Calibri",align:"center"});
    s.addText("Bakış Açısı", {x:x+0.12,y:3.4,w:3.81,h:0.28, fontSize:9,color:C.gray,bold:true,fontFace:"Calibri"});
    s.addText(perspective, {x:x+0.12,y:3.68,w:3.81,h:0.55, fontSize:11,color:C.navy,fontFace:"Calibri",italic:true});
    s.addShape(pres.shapes.RECTANGLE, {x:x+0.12,y:4.28,w:3.81,h:0.01, fill:{color:C.grayLt}, line:{color:C.grayLt}});
    s.addText("✓ "+strength, {x:x+0.12,y:4.32,w:3.81,h:0.85, fontSize:10,color:C.green,fontFace:"Calibri",valign:"top"});
    s.addText("⚠ "+weakness, {x:x+0.12,y:5.22,w:3.81,h:0.75, fontSize:10,color:C.amber,fontFace:"Calibri",valign:"top"});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:x+0.12,y:6.02,w:3.81,h:0.38, fill:{color}, line:{color}, rounding:0.05});
    s.addText("Ağırlık: "+weight, {x:x+0.12,y:6.02,w:3.81,h:0.38, fontSize:9,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — FEATURE-LEVEL FUSION: NEDEN VE NASIL KARAR VERİYOR?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Feature-Level Fusion — Neden Bu Formülü Seçtik?", "Oylamada kaynak çeşitliliğini ödüllendiriyoruz");
  badge(s,12);

  // Problem statement
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:1.42,w:12.7,h:0.68,
    fill:{color:"FFF8F0"}, line:{color:C.amber}, rounding:0.07,
  });
  s.addText("Sorun: 6 sinyal birden fazla 'candidate boundary' üretir ve bunlar birbirinden birkaç yüz milisaniye farklı olabilir. Hangisi gerçek?  —  Bağımsız kaynak sayısı arttıkça güven artar.", {
    x:0.5,y:1.42,w:12.3,h:0.68, fontSize:12,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle",
  });

  // 4 design decisions
  const decisions = [
    {
      d:"Neden temporal gruplama?",
      body:"SSM 30.8 s, RMS 31.5 s diyor. İkisi de aynı geçişi gördü — sadece farklı frame'de tespit etti. 2.75 s pencere içindeki adayları tek grup sayıyoruz.",
      color:C.copper,
    },
    {
      d:"Neden her kaynaktan tek oy?",
      body:"SSM gürültülüyse aynı bölgede 10 peak üretebilir. Tek kaynaktan yalnızca en yüksek confidence'lı aday alınır. Böylece bir kaynak oylamayı domine edemez.",
      color:C.teal,
    },
    {
      d:"Neden sadece ağırlıklı toplam değil, agreement bonus?",
      body:"Yalnızca SSM mi diyor? O zaman score = 0.42 × conf. Ama 3 bağımsız sinyal aynı noktayı gösteriyorsa bonus ekliyoruz (max 0.15). Consensus güvenilirliği artırır.",
      color:C.green,
    },
    {
      d:"Neden SSM tek başına kabul ediliyor?",
      body:"SSM yapısal tekrar bilgisi taşıyan tek sinyaldir. Diğer sinyaller zayıfsa dahi güçlü bir SSM candidate'ını (conf ≥ 0.5) boundary olarak tutuyoruz. Yoksa gerçek section boundary'leri kaçırma riski çok yüksek.",
      color:C.amber,
    },
  ];

  decisions.forEach(({d,body,color},i) => {
    const col=i%2, row=Math.floor(i/2);
    const x=0.3+col*6.55, y=2.3+row*2.35;
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:2.2, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:0.07,h:2.2, fill:{color}, line:{color}});
    s.addText(d, {x:x+0.2,y:y+0.1,w:5.8,h:0.45, fontSize:13,color,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(body, {x:x+0.2,y:y+0.6,w:5.8,h:1.5, fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });

  // Formula strip
  s.addShape(pres.shapes.RECTANGLE, {x:0.3,y:7.05,w:12.7,h:0.42, fill:{color:C.navyMid}, line:{color:C.navyMid}});
  s.addText("score = Σ(kaynak_ağırlığı × candidate_confidence)  +  min(0.15,  0.035 × (kaynak_sayısı − 1))     →     kabul: score ≥ 0.30  VEYA  güçlü SSM", {
    x:0.5,y:7.05,w:12.3,h:0.42, fontSize:10.5,color:C.copperLt,bold:true,fontFace:"Calibri",align:"center",valign:"middle",
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — NEDEN İKİNCİ FUSION SEVİYESİ?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Neden İki Seviye Fusion?", "Feature fusion yeterli değil — algorithm-level fusion neden gerekiyor?");
  badge(s,13);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"F0FDF4"}, line:{color:C.green}, rounding:0.07,
  });
  s.addText("Feature-level fusion tek bir algoritmanın içindedir. Custom_librosa'nın verdiği sonuç zaten 6 sinyali birleştirdi. Ama custom_librosa'nın kendisi de bazen hata yapar. İkinci fusion seviyesi farklı algoritmaların hatalarını dengeler.", {
    x:0.5,y:1.42,w:12.3,h:0.72, fontSize:12,color:C.green,fontFace:"Calibri",align:"left",valign:"middle",
  });

  // Two-level diagram
  s.addText("Seviye 1 — Feature Fusion (custom_librosa içinde):", {
    x:0.3,y:2.32,w:12.7,h:0.35, fontSize:13,color:C.navy,bold:true,fontFace:"Georgia",
  });
  flowRow(s, [
    {label:"RMS\nsinyal",color:C.copper},
    {label:"Chroma\nsinyal",color:C.teal},
    {label:"Onset\nsinyal",color:C.green},
    {label:"SSM\nsinyal",color:C.amber},
    {label:"Beat\nsinyal",color:C.purple},
  ], 0.5, 2.73, 1.8, 0.85);
  s.addText("→", {x:10.2,y:2.88,w:0.3,h:0.5, fontSize:18,color:C.copper,bold:true,align:"center"});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:10.55,y:2.73,w:2.5,h:0.85, fill:{color:C.copper},line:{color:C.copper},rounding:0.08});
  s.addText("custom_librosa\nsonucu", {x:10.55,y:2.73,w:2.5,h:0.85, fontSize:11,color:C.white,bold:true,align:"center",valign:"middle"});

  s.addShape(pres.shapes.RECTANGLE, {x:0.3,y:3.73,w:12.7,h:0.01, fill:{color:C.grayLt}, line:{color:C.grayLt}});

  s.addText("Seviye 2 — Algorithm Fusion (ayrı worker):", {
    x:0.3,y:3.82,w:12.7,h:0.35, fontSize:13,color:C.navy,bold:true,fontFace:"Georgia",
  });
  flowRow(s, [
    {label:"custom_librosa\nsonucu",color:C.copper},
    {label:"Foote\nsonucu",color:"5B21B6"},
    {label:"CNMF\nsonucu",color:C.teal},
    {label:"SCluster\nsonucu",color:C.green},
  ], 0.5, 4.23, 2.3, 0.85);
  s.addText("→", {x:10.2,y:4.38,w:0.3,h:0.5, fontSize:18,color:C.copper,bold:true,align:"center"});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:10.55,y:4.23,w:2.5,h:0.85, fill:{color:C.navy},line:{color:C.navy},rounding:0.08});
  s.addText("FUSION\nsonucu", {x:10.55,y:4.23,w:2.5,h:0.85, fontSize:11,color:C.white,bold:true,align:"center",valign:"middle"});

  // Why two levels is the right choice
  const table = [
    ["","Feature-Level Fusion","Algorithm-Level Fusion"],
    ["Oy verenler","Sinyal kaynakları (SSM, RMS, onset…)","Bağımsız segmentasyon algoritmaları"],
    ["Girdi","Ham audio feature candidate'ları","Tamamlanmış AlgorithmResult nesneleri"],
    ["Konum","custom_librosa pipeline içinde","Ayrı FusionWorker servisi"],
    ["Amaç","Tek pipeline içinde sinyal çeşitliliği","Farklı hata profillerini dengeleme"],
  ];
  s.addTable(table.map((row,ri) => row.map((cell,ci) => ({
    text: cell,
    options: {
      fontSize: ri===0||ci===0 ? 11 : 10.5,
      bold: ri===0||ci===0,
      color: ri===0 ? C.white : ci===0 ? C.navy : C.slate,
      fill: ri===0 ? {color:C.navy} : ci===0 ? {color:"EEF2FF"} : {color:C.white},
      align:"center",
    },
  }))), {x:0.3,y:5.28,w:12.7,h:2.05, colW:[2.5,5.1,5.1], border:{pt:1,color:C.grayLt}});
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — SECTION DIVIDER
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  divider(s, "Algorithm-Level Fusion", "Neden ağırlıklı oy? Neden tüm sonuçları bekliyoruz?");
  badge(s,14);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — NEDEN AĞIRLIKLI OY?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Neden Ağırlıklı Oy? — Basit Çoğunluk Yetmez", "Algoritmalar eşit güvenilir değil; confidence bilgisi de önemli");
  badge(s,15);

  const alternatives = [
    {
      title:"❌ Simple Majority Vote",
      problems:[
        "Her algoritmayı eşit güvenilir sayar",
        "Confidence değerini görmezden gelir",
        "2/4 oyu olan boundary kabul edilir — zayıf kanıt",
        "Over-segmenting bir algoritma fazla boundary üretip galip gelebilir",
      ],
      color:C.red,
    },
    {
      title:"❌ Simple Average (zaman ortalaması)",
      problems:[
        "Aynı boundary'yi bulan algoritmaları gruplamaz",
        "10 s arayla iki farklı boundary'nin ortası anlamsız",
        "Confidence ve ağırlık bilgisini kullanmaz",
      ],
      color:C.amber,
    },
    {
      title:"✓ Weighted Voting (seçtiğimiz)",
      problems:[
        "score = Σ(algoritmik_ağırlık × boundary_confidence)",
        "custom_librosa (0.35) > scluster (0.30) > cnmf (0.20) > foote (0.15)",
        "Grup içinde her algoritma en fazla 1 oy verir",
        "Kabul: score ≥ threshold  VEYA  kaynak_sayısı ≥ gerekli_oy",
      ],
      color:C.green,
    },
  ];

  alternatives.forEach(({title,problems,color},i) => {
    const x=0.3+i*4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x,y:1.42,w:4.05,h:4.5,
      fill:{color:C.white}, line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.08}, rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE, {x,y:1.42,w:4.05,h:0.58, fill:{color}, line:{color}});
    s.addText(title, {x:x+0.1,y:1.42,w:3.85,h:0.58, fontSize:12,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
    bullets(s, problems, x+0.1, 2.05, 3.85, 3.8, 11, C.slate);
  });

  // Why two acceptance conditions
  s.addShape(pres.shapes.RECTANGLE, {x:0.3,y:6.1,w:12.7,h:0.42, fill:{color:C.navyMid}, line:{color:C.navyMid}});
  s.addText("Neden iki kabul koşulu (score VEYA oy sayısı)?", {x:0.3,y:6.1,w:12.7,h:0.42, fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
  const condExplain = [
    ["Tek threshold kullanırsak","Confidence calibration algoritmalar arasında tam değilse yüksek ağırlıklı algoritma sonucu körü körüne kabul edebilir"],
    ["Oy sayısı fallback'i ile","İki bağımsız algoritma aynı bölgeyi gösterdiyse — confidence düşük bile olsa — consensus korunur"],
  ];
  condExplain.forEach(([t,d],i) => {
    s.addText("▸ "+t+": "+d, {x:0.4,y:6.6+i*0.35,w:12.5,h:0.3, fontSize:10.5,color:C.grayLt,fontFace:"Calibri",align:"left",valign:"middle"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — NEDEN TÜM SONUÇLARI BEKLİYORUZ? (fusion orchestration)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Fusion Neden Tüm Base Sonuçları Bekliyor?", "Yarım bilgiyle fusion yapmak sonucu kötüleştirir");
  badge(s,16);

  // Main reasoning
  const reasons = [
    {
      title:"Erken fusion yanlı sonuç üretir",
      body:"Foote hızlı bitip custom_librosa yavaş kalırsa: yalnızca Foote verisiyle fusion yapılır. Sonuç sanki 4 algoritma konuştu gibi görünür ama aslında 1 algoritmanın sesi vardır. Bu yanlış güven verir.",
      color:C.red,
    },
    {
      title:"Tüm perspektifler gelince consensus anlamlı olur",
      body:"Custom_librosa yapısal SSM, SCluster global clustering, CNMF latent pattern, Foote local novelty — bunların dördü birden aynı boundary'yi gösterince güven gerçekten yükselir.",
      color:C.green,
    },
    {
      title:"En az 2 başarılı sonuç şartı",
      body:"4 base'in tamamı resolved olduğunda başarılı sonuç 2'nin altındaysa fusion işe yaramaz. ResultListener doğrudan 'failed fusion' sonucu üretir — gereksiz hesaplama yapmaz.",
      color:C.teal,
    },
    {
      title:"Bilinen sınır: worker kaybolursa ne olur?",
      body:"Worker exception fırlatırsa BaseWorker 'failed' sonuç yayınlar → resolved sayılır. Ama worker hiç yanıt vermeden çöküp giderse task processing'de kalır. Watchdog/timeout gelecek çalışma.",
      color:C.amber,
    },
  ];

  reasons.forEach(({title,body,color},i) => {
    const col=i%2, row=Math.floor(i/2);
    const x=0.3+col*6.55, y=1.42+row*2.88;
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:2.68, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:0.07,h:2.68, fill:{color}, line:{color}});
    s.addText(title, {x:x+0.2,y:y+0.1,w:5.8,h:0.55, fontSize:13,color,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(body, {x:x+0.2,y:y+0.72,w:5.8,h:1.9, fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });

  quoteBar(s,"Güven için bütünlük şart: 4 farklı bakış açısının tamamından consensus alıyoruz, yoksa fusion yapmıyoruz",7.24);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 17 — NEDEN YAPISAL VE SEMANTİK LABEL AYRI?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Neden İki Katmanlı Etiketleme?", "Benzerlik iddiası ile müzikal rol iddiası farklı güç gerektirir");
  badge(s,17);

  // Core argument
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"FFF8F0"}, line:{color:C.amber}, rounding:0.07,
  });
  s.addText("'Bu iki segment birbirine benziyor' demek ile 'bu segment Chorus' demek aynı iddianın gücünde değil. Birincisi audio descriptor ölçümüdür. İkincisi müzikal yorumdur.", {
    x:0.5,y:1.42,w:12.3,h:0.72, fontSize:13,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle",
  });

  // Visual: label example
  const segs = [
    {sl:"A",sem:"Intro",c:C.copper},{sl:"B",sem:"Verse",c:C.teal},{sl:"C",sem:"Chorus",c:C.green},
    {sl:"B",sem:"Verse",c:C.teal},{sl:"C",sem:"Chorus",c:C.green},{sl:"D",sem:"Bridge",c:C.amber},{sl:"A",sem:"Outro",c:C.copper},
  ];
  const w = 13.3/segs.length;
  segs.forEach(({sl,sem,c},i) => {
    s.addShape(pres.shapes.RECTANGLE, {x:i*w,y:2.32,w:w-0.03,h:0.6, fill:{color:c}, line:{color:c}});
    s.addText(`Struct: ${sl}`, {x:i*w,y:2.32,w:w-0.03,h:0.32, fontSize:10,color:C.white,bold:true,align:"center",valign:"middle"});
    s.addText(sem, {x:i*w,y:2.64,w:w-0.03,h:0.28, fontSize:9,color:C.white,align:"center",valign:"middle"});
  });

  const layers = [
    {
      title:"Structural Label (A/B/C)",
      subtitle:"'Bu segment o segmente benziyor' — ölçülebilir, güvenilir",
      points:[
        "Chroma, MFCC, RMS, onset density descriptor'larından cluster edilir",
        "Agglomerative clustering; silhouette score ile en iyi k seçilir",
        "A = en sık küme, B = ikinci, sırayla — Verse anlamına gelmez",
        "Dürüst iddia: audio descriptor benzerliği ölçüldü",
      ],
      color:C.copper,
    },
    {
      title:"Semantic Label (Intro/Verse/Chorus…)",
      subtitle:"'Bu section müzikal olarak X rolünü oynuyor' — heuristic, daha zayıf",
      points:[
        "Heuristic kurallar: ilk %20'deki unique segment → Intro tahmini",
        "Tekrar eden, daha yüksek enerjili cluster → Chorus tahmini",
        "Structural label'ı overwrite etmez — ayrı alan",
        "Her label için semantic_confidence ve semantic_reason yazılır",
        "Kanıt yetersizse 'Unknown' veya 'Early/Middle/Late' verilir",
      ],
      color:C.teal,
    },
  ];

  layers.forEach(({title,subtitle,points,color},i) => {
    const x=0.3+i*6.55;
    s.addShape(pres.shapes.RECTANGLE, {x,y:3.12,w:6.1,h:4.28, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE, {x,y:3.12,w:6.1,h:0.55, fill:{color}, line:{color}});
    s.addText(title, {x:x+0.1,y:3.12,w:5.9,h:0.55, fontSize:14,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
    s.addText(subtitle, {x:x+0.1,y:3.72,w:5.9,h:0.45, fontSize:11,color,bold:true,fontFace:"Calibri",italic:true,align:"left",valign:"middle"});
    bullets(s, points, x+0.1, 4.22, 5.9, 3.1, 11, C.slate);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 18 — NEDEN İKİ FARKLI TOLERANS?
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Evaluation — Neden İki Farklı Tolerans?", "Bir ölçüm sistemin hangi sorununu saklıyor?");
  badge(s,18);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"EFF6FF"}, line:{color:C.teal}, rounding:0.07,
  });
  s.addText("Tek toleransla ölçüm yapmak bir sistemin gerçek performansını gizleyebilir. Doğru bölgeyi bulan ama tam saniyeyi kaçıran sistemle hem bölgeyi hem saniyeyi kaçıran sistemi ayırt edemezsiniz.", {
    x:0.5,y:1.42,w:12.3,h:0.72, fontSize:12,color:C.navy,fontFace:"Calibri",align:"left",valign:"middle",
  });

  // Two tolerance explanations
  const tols = [
    {
      tol:"±0.5 saniye — Strict",
      question:"Tam zamanı bulduk mu?",
      measures:"Boundary timestamp hassasiyetini ölçer. F1@0.5 düşükse algoritma yapısal geçişin bölgesini buluyor ama exact millisecond'u kaçırıyor olabilir.",
      lowMeans:"Snapping, frame çözünürlüğü veya SSM smoothing timestamp'i kaydırıyor",
      color:C.copper,
    },
    {
      tol:"±3.0 saniye — Lenient",
      question:"Doğru bölgeyi fark ettik mi?",
      measures:"Yapısal region detection'ı ölçer. Bir section'ın var olduğunu fark ettik mi? F1@3.0 yüksekse sistem müzikal yapıyı anlıyor demektir.",
      lowMeans:"Temel boundary detection veya boundary sayısı problemi var",
      color:C.teal,
    },
  ];

  tols.forEach(({tol,question,measures,lowMeans,color},i) => {
    const x=0.3+i*6.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x,y:2.32,w:6.1,h:3.65,
      fill:{color:C.white}, line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.07}, rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE, {x,y:2.32,w:6.1,h:0.65, fill:{color}, line:{color}});
    s.addText(tol, {x:x+0.1,y:2.32,w:5.9,h:0.65, fontSize:16,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
    s.addText("Soru: "+question, {x:x+0.1,y:3.03,w:5.9,h:0.38, fontSize:13,color,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText(measures, {x:x+0.1,y:3.47,w:5.9,h:1.05, fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:x+0.1,y:4.6,w:5.9,h:1.22, fill:{color:"FFF7ED"}, line:{color:C.amber}, rounding:0.05});
    s.addText("Düşükse: "+lowMeans, {x:x+0.2,y:4.65,w:5.7,h:1.12, fontSize:11,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle"});
  });

  // Interpretation table
  s.addShape(pres.shapes.RECTANGLE, {x:0.3,y:6.1,w:12.7,h:0.42, fill:{color:C.navyMid}, line:{color:C.navyMid}});
  s.addText("Sonuç yorumlama kılavuzu", {x:0.3,y:6.1,w:12.7,h:0.42, fontSize:12,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});

  const interp = [
    ["F1@3 yüksek,  F1@0.5 düşük","Doğru bölgeyi buluyor ama timestamp hassasiyeti zayıf → snapping geliştirilebilir"],
    ["Her ikisi de yüksek","Hem bölge hem timing doğru → iyi çalışan sistem"],
    ["Her ikisi de düşük","Temel boundary detection problemi var → boundary sayısı veya konumu hatalı"],
  ];
  interp.forEach(([scenario,meaning],i) => {
    const y=6.6+i*0.38;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:0.3,y,w:4.5,h:0.32, fill:{color:C.copper},line:{color:C.copper},rounding:0.05});
    s.addText(scenario, {x:0.3,y,w:4.5,h:0.32, fontSize:9.5,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
    s.addText("→  "+meaning, {x:4.9,y,w:8.2,h:0.32, fontSize:10.5,color:C.grayLt,fontFace:"Calibri",align:"left",valign:"middle"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 19 — DEMO
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Canlı Demo", "Sistemin kendi diagnostics'iyle kararlarını açıklaması");
  badge(s,19);

  const steps = [
    {n:"1",title:"Track seç, 4 base + fusion başlat",
     why:"Request gönderildiğinde fusion worker hemen başlamaz — önce 4 bağımsız worker paralel dispatch edilir. RabbitMQ ayrı routing key'lere mesaj iletir.",color:C.copper},
    {n:"2",title:"Partial sonuçlar gelirken status izle",
     why:"Task hâlâ 'processing'. Custom ve Foote geldi ama CNMF ve SCluster bekleniyor. Frontend SSE stream'i ile anlık güncelleniyor.",color:C.teal},
    {n:"3",title:"Her algoritmanın farklı boundary'lerini karşılaştır",
     why:"İşte neden fusion gerekti bunun göstergesi: 4 algoritmanın boundary timeline'larının birbirinden farklı olduğunu canlı görebilirsiniz.",color:C.green},
    {n:"4",title:"Fusion diagnostics'i aç",
     why:"Her fused boundary hangi algoritmalar oy verdi, ham timestamp neydi, weighted score neydi, kabul mi reddedildi mi — hepsi açıklanmış durumda.",color:C.amber},
    {n:"5",title:"Evaluation sonuçlarını göster",
     why:"F1@0.5 ve F1@3.0 değerleri. Strict ve lenient toleransı beraber yorumla: bölge mi doğru, timing mi doğru, ikisi birden mi?",color:C.purple},
  ];

  steps.forEach(({n,title,why,color},i) => {
    const y=1.42+i*1.12;
    s.addShape(pres.shapes.RECTANGLE, {x:0.3,y,w:12.7,h:1.0, fill:{color:C.white}, line:{color:C.grayLt}});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {x:0.35,y:y+0.1,w:0.55,h:0.55, fill:{color}, line:{color}, rounding:0.08});
    s.addText(n, {x:0.35,y:y+0.1,w:0.55,h:0.55, fontSize:15,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
    s.addText(title, {x:1.05,y:y+0.08,w:5.5,h:0.4, fontSize:13,color:C.navy,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText("Anlatılacak: "+why, {x:1.05,y:y+0.52,w:11.8,h:0.42, fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x:0.3,y:7.05,w:12.7,h:0.4,
    fill:{color:"FFF7ED"}, line:{color:C.amber}, rounding:0.05,
  });
  s.addText("Risk planı: Önceden tamamlanmış bir task ID ve fusion diagnostics JSON'unun screenshot'ı hazır tutun. Worker gecikirerse bunu asenkron mimarinin avantajını göstermek için kullanın.", {
    x:0.5,y:7.05,w:12.3,h:0.4, fontSize:10,color:C.amber,bold:true,fontFace:"Calibri",align:"center",valign:"middle",
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 20 — SONUÇ
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  header(s, "Sonuç — Ne Yaptık ve Neden?", "Her kararın arkasında bir problem bağlamı var");
  badge(s,20);

  const decisions = [
    {d:"Dağıtık mimari",why:"DSP işlemi API'yi bloke etmesin, her algoritma bağımsız scale edilsin",color:C.teal},
    {d:"Ortak result şeması",why:"Fusion hiçbir algoritmayı özel olarak tanımasın; yeni worker eklemek sadece şemaya uymak olsun",color:C.copper},
    {d:"Feature-level fusion",why:"Tek sinyal her genre'da güvenilir değil; 6 bağımsız kanıtın consensus'u daha sağlam",color:C.green},
    {d:"Algorithm-level fusion",why:"Tek pipeline da hata yapabilir; farklı matematiksel perspektiflerden gelen oylar hata profillerini dengeler",color:C.amber},
    {d:"Yapısal/semantik ayrımı",why:"Benzerlik iddiası ölçülebilir; müzikal rol iddiası heuristic — ikisini karıştırmak bilimsel dürüstlüğü bozar",color:C.purple},
    {d:"İki tolerans (0.5s / 3.0s)",why:"Bölgeyi bulmak ile tam saniyeyi bulmak farklı beceriler — tek metrik hangisinde kötü olduğumuzu gizler",color:C.navy},
  ];

  decisions.forEach(({d,why,color},i) => {
    const col=i%2, row=Math.floor(i/2);
    const x=0.3+col*6.55, y=1.42+row*1.82;
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:1.65, fill:{color:C.white}, line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:4,offset:1,angle:135,opacity:0.07}});
    s.addShape(pres.shapes.RECTANGLE, {x,y,w:6.1,h:0.5, fill:{color}, line:{color}});
    s.addText("✓ "+d, {x:x+0.1,y,w:5.9,h:0.5, fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText("Neden: "+why, {x:x+0.1,y:y+0.56,w:5.9,h:1.05, fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });

  // Closing
  s.addShape(pres.shapes.RECTANGLE, {x:0,y:7.38,w:13.3,h:0.42, fill:{color:C.navy}, line:{color:C.navy}});
  s.addText("Katkımız yeni bir algoritma değil — çoklu sinyal, çoklu algoritma ve iki seviye fusion kararlarının dağıtık, açıklanabilir ve ölçülebilir bir sistemde birleştirilmesidir.", {
    x:0.3,y:7.38,w:12.7,h:0.42, fontSize:10.5,color:C.copperLt,fontFace:"Calibri",align:"center",valign:"middle",
  });
}

// ─── GENERATE ────────────────────────────────────────────────────────────────
pres.writeFile({fileName:"docs/MusicSegmentation_FinalPresentation.pptx"})
  .then(()=>console.log("✅  docs/MusicSegmentation_FinalPresentation.pptx oluşturuldu"))
  .catch(e=>{console.error("❌ Hata:",e);process.exit(1);});
