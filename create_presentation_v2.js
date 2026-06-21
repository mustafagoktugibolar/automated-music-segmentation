"use strict";
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout  = "LAYOUT_WIDE";
pres.title   = "Automated Music Structure Segmentation";
pres.author  = "Capstone Team";

// ─── PALETTE ─────────────────────────────────────────────────────────────────
const C = {
  navy:"1B2A4A", navyMid:"243556",
  copper:"B87333", copperLt:"D4944A",
  slate:"4A5568", white:"FFFFFF", offWhite:"F4F6FA",
  gray:"718096",  grayLt:"E2E8F0",
  green:"2D6A4F", greenLt:"52B788",
  amber:"B45309", teal:"0D9488",
  red:"9B1C1C",  purple:"6D28D9",
};

// ─── PRESENTER MAP ────────────────────────────────────────────────────────────
// 21 slides: P1=1-7, P2=8-14, P3=15-21
const PM = {};
for(let i=1;i<=7;i++)  PM[i]="Presenter 1";
for(let i=8;i<=14;i++) PM[i]="Presenter 2";
for(let i=15;i<=21;i++)PM[i]="Presenter 3";
const PC = {"Presenter 1":C.copper,"Presenter 2":C.teal,"Presenter 3":C.green};

// ─── GLOBAL HELPERS ───────────────────────────────────────────────────────────
function badge(s,n){ const w=PM[n],col=PC[w];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:10.5,y:7.0,w:2.6,h:0.34,fill:{color:col},line:{color:col},rounding:0.08});
  s.addText(`🎤 ${w}`,{x:10.5,y:7.0,w:2.6,h:0.34,fontSize:9,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
  s.addText(`${n}/21`,{x:0.2,y:7.05,w:0.9,h:0.22,fontSize:8,color:C.gray,align:"left",margin:0});
}

function hdr(s,title,sub){
  s.background={color:C.offWhite};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:13.3,h:1.15,fill:{color:C.navy},line:{color:C.navy}});
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:1.15,w:13.3,h:0.055,fill:{color:C.copper},line:{color:C.copper}});
  s.addText(title,{x:0.4,y:0.08,w:11.5,h:0.66,fontSize:26,color:C.white,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
  if(sub) s.addText(sub,{x:0.4,y:0.73,w:11.5,h:0.38,fontSize:13,color:C.copperLt,fontFace:"Calibri",align:"left",valign:"middle",margin:0});
}

function div(s,title,sub){
  s.background={color:C.navy};
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:2.9,w:13.3,h:0.06,fill:{color:C.copper},line:{color:C.copper}});
  s.addText(title,{x:1,y:1.7,w:11.3,h:1.2,fontSize:40,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
  if(sub) s.addText(sub,{x:1.5,y:3.2,w:10.3,h:0.9,fontSize:16,color:C.copperLt,fontFace:"Calibri",align:"center"});
}

function qbar(s,txt,y){
  s.addShape(pres.shapes.RECTANGLE,{x:0,y,w:13.3,h:0.46,fill:{color:C.navy},line:{color:C.navy}});
  s.addText(txt,{x:0.4,y,w:12.5,h:0.46,fontSize:11,color:C.copperLt,fontFace:"Calibri",align:"center",valign:"middle"});
}

function bullets(s,items,x,y,w,h,fs,col){
  const rich=items.map((t,i)=>({text:t,options:{bullet:true,breakLine:i<items.length-1,fontSize:fs||13,color:col||C.slate,fontFace:"Calibri"}}));
  s.addText(rich,{x,y,w,h,valign:"top",margin:[6,10,6,10]});
}

function card(s,title,body,x,y,w,h,ac){
  const c=ac||C.copper;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h,fill:{color:C.white},line:{color:C.grayLt},
    shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.08},rounding:0.08});
  s.addShape(pres.shapes.RECTANGLE,{x,y,w,h:0.055,fill:{color:c},line:{color:c}});
  s.addText(title,{x:x+0.12,y:y+0.1,w:w-0.24,h:0.36,fontSize:12,bold:true,color:C.navy,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
  bullets(s,body,x+0.12,y+0.5,w-0.24,h-0.6,11,C.slate);
}

// 3-row reason card: Problem → Decision → Why it works
function rcCard(s,x,y,w,h,problem,decision,why,ac){
  const c=ac||C.copper;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h,fill:{color:C.white},line:{color:C.grayLt},
    shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.08},rounding:0.08});
  s.addShape(pres.shapes.RECTANGLE,{x,y,w,h:0.055,fill:{color:c},line:{color:c}});
  const rh=(h-0.1)/3;
  [["PROBLEM",problem,C.red],["DECISION",decision,C.navy],["WHY IT WORKS",why,C.green]].forEach(([lbl,val,col],i)=>{
    const iy=y+0.12+i*rh;
    s.addText(lbl,{x:x+0.14,y:iy,w:w-0.28,h:0.22,fontSize:8,color:col,bold:true,fontFace:"Calibri",align:"left",valign:"middle",margin:0});
    s.addText(val,{x:x+0.14,y:iy+0.22,w:w-0.28,h:rh-0.28,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top",margin:0});
    if(i<2) s.addShape(pres.shapes.RECTANGLE,{x:x+0.14,y:iy+rh-0.04,w:w-0.28,h:0.01,fill:{color:C.grayLt},line:{color:C.grayLt}});
  });
}

function flowRow(s,items,x,y,iw,ih){
  items.forEach(({label,color,tc},i)=>{
    const ix=x+i*(iw+0.22);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:ix,y,w:iw,h:ih,fill:{color:color||C.navy},line:{color:color||C.navy},rounding:0.09});
    s.addText(label,{x:ix,y,w:iw,h:ih,fontSize:10,color:tc||C.white,bold:true,fontFace:"Calibri",align:"center",valign:"middle"});
    if(i<items.length-1) s.addText("→",{x:ix+iw,y:y+ih/2-0.15,w:0.22,h:0.3,fontSize:13,color:C.copper,bold:true,align:"center"});
  });
}

function vsBox(s,lT,lI,rT,rI,x,y,w,h,lc,rc){
  const cw=w/2-0.05,ll=lc||C.green,rr=rc||C.red;
  [[lT,lI,ll,x],[rT,rI,rr,x+cw+0.1]].forEach(([t,items,col,bx])=>{
    s.addShape(pres.shapes.RECTANGLE,{x:bx,y,w:cw,h,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE,{x:bx,y,w:cw,h:0.45,fill:{color:col},line:{color:col}});
    s.addText(t,{x:bx,y,w:cw,h:0.45,fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
    bullets(s,items,bx+0.1,y+0.52,cw-0.2,h-0.6,11,C.slate);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  s.background={color:C.navy};
  s.addShape(pres.shapes.OVAL,{x:10.8,y:-0.6,w:4,h:4,fill:{color:"1F3460"},line:{color:"1F3460"}});
  s.addShape(pres.shapes.OVAL,{x:-1.2,y:5.5,w:3,h:3,fill:{color:"162440"},line:{color:"162440"}});
  s.addShape(pres.shapes.RECTANGLE,{x:0,y:4.6,w:13.3,h:0.07,fill:{color:C.copper},line:{color:C.copper}});
  s.addText("Automated Music",{x:0.7,y:0.9,w:11,h:1.0,fontSize:46,color:C.white,bold:true,fontFace:"Georgia",align:"center"});
  s.addText("Structure Segmentation",{x:0.7,y:1.85,w:11,h:1.0,fontSize:46,color:C.copperLt,bold:true,fontFace:"Georgia",align:"center"});
  s.addText("Why multi-algorithm, why two fusion levels, why distributed architecture?",{
    x:1,y:3.0,w:11.3,h:0.55,fontSize:14,color:"CBD5E0",fontFace:"Calibri",align:"center"});
  s.addText("Capstone Final Presentation  •  Spring 2026",{
    x:1,y:3.62,w:11.3,h:0.4,fontSize:12,color:C.gray,fontFace:"Calibri",align:"center"});
  ["Presenter 1","Presenter 2","Presenter 3"].forEach((p,i)=>{
    const col=[C.copper,C.teal,C.green][i];
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:2.4+i*3.2,y:4.9,w:2.8,h:0.4,fill:{color:col},line:{color:col},rounding:0.1});
    s.addText(p,{x:2.4+i*3.2,y:4.9,w:2.8,h:0.4,fontSize:12,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
  });
  badge(s,1);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — PRESENTATION LOGIC
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Presentation Logic","Each section answers the question raised by the previous one");
  badge(s,2);

  const chain=[
    ["01","What is the problem?","Detecting structural transition timestamps in a music track",C.copper],
    ["02","Why is it hard?","No single signal reliably reveals all section boundaries",C.teal],
    ["03","What kind of system?","Distributed, asynchronous — each algorithm runs independently",C.green],
    ["04","Isn't one algorithm enough?","6 signals are fused inside custom_librosa: feature-level fusion",C.amber],
    ["05","One pipeline still not enough?","Different algorithms have different error profiles — algorithm-level fusion balances them",C.purple],
    ["06","How do we know it works?","SALAMI annotations + two tolerances (0.5 s / 3.0 s) with full dataset results",C.navy],
  ];

  chain.forEach(([num,q,a,color],i)=>{
    const col=i%2, row=Math.floor(i/2);
    const x=0.3+col*6.55, y=1.45+row*1.9;
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:1.7,fill:{color:C.white},line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:4,offset:1,angle:135,opacity:0.07}});
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:0.07,h:1.7,fill:{color},line:{color}});
    s.addText(num,{x:x+0.2,y:y+0.08,w:0.6,h:0.6,fontSize:22,color,bold:true,fontFace:"Georgia",align:"center",margin:0});
    s.addText("Q: "+q,{x:x+0.85,y:y+0.1,w:5.1,h:0.42,fontSize:12,color:C.navy,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText("A: "+a,{x:x+0.85,y:y+0.58,w:5.1,h:0.95,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — PROBLEM: BOUNDARY DETECTION
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Problem: Boundary Detection","Finding the exact timestamps where musical structure changes");
  badge(s,3);

  // Timeline
  const segs=[{l:"Intro",c:C.copper,w:2.0},{l:"Verse",c:C.teal,w:2.7},{l:"Chorus",c:C.green,w:2.4},
               {l:"Verse",c:C.teal,w:2.4},{l:"Chorus",c:C.green,w:2.4},{l:"Outro",c:C.amber,w:1.3}];
  let cx=0.3;
  segs.forEach(({l,c,w})=>{
    s.addShape(pres.shapes.RECTANGLE,{x:cx,y:1.45,w,h:0.85,fill:{color:c},line:{color:c}});
    s.addText(l,{x:cx,y:1.45,w,h:0.85,fontSize:12,color:C.white,bold:true,align:"center",valign:"middle"});
    cx+=w;
  });
  [2.3,5.0,7.4,9.8].forEach(x=>{
    s.addShape(pres.shapes.RECTANGLE,{x:x+0.3,y:1.2,w:0.04,h:1.35,fill:{color:"DC2626"},line:{color:"DC2626"}});
    s.addText("◀ boundary",{x:x,y:2.6,w:1.0,h:0.25,fontSize:8,color:"DC2626",align:"center"});
  });

  const pts=[
    {icon:"🎯",t:"Primary task: boundary, NOT the label",
     b:"The computer's first challenge is not naming a section — it is finding the exact second where the transition occurs. Labeling comes after.",c:C.copper},
    {icon:"📏",t:"Segment = interval between two boundaries",
     b:"Once two consecutive boundaries are found, the time interval between them is a segment. Segment count follows boundary count.",c:C.teal},
    {icon:"⚖️",t:"Too many or too few boundaries — both are errors",
     b:"Over-segmentation: unnecessary extra cuts → precision drops. Under-segmentation: missed transitions → recall drops.",c:C.green},
  ];
  pts.forEach(({icon,t,b,c},i)=>{
    const x=0.3+i*4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:3.05,w:4.05,h:3.55,
      fill:{color:C.white},line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.07},rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE,{x,y:3.05,w:4.05,h:0.06,fill:{color:c},line:{color:c}});
    s.addText(icon,{x:x+0.15,y:3.15,w:0.55,h:0.55,fontSize:24,align:"center",valign:"middle"});
    s.addText(t,{x:x+0.75,y:3.15,w:3.15,h:0.55,fontSize:12,color:c,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText(b,{x:x+0.15,y:3.78,w:3.75,h:2.7,fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — WHY IS IT HARD?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Why Is It Hard? — No Single Signal Is Enough","Every musical transition leaves a different acoustic footprint");
  badge(s,4);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"EFF6FF"},line:{color:C.teal},rounding:0.07});
  s.addText("Core contradiction: a Verse → Chorus transition sometimes raises energy, sometimes shifts chords, sometimes only adds drums — none of these always happen.",{
    x:0.5,y:1.42,w:12.3,h:0.72,fontSize:13,color:C.navy,fontFace:"Calibri",align:"left",valign:"middle"});

  const examples=[
    {sig:"Only RMS (energy)",prob:"Energy rises at Verse→Chorus, but two verses with an acoustic guitar sound identical in loudness",verdict:"INSUFFICIENT",c:C.red},
    {sig:"Only Chroma (harmony)",prob:"Two different sections sharing the same chord progression look harmonically identical",verdict:"INSUFFICIENT",c:C.red},
    {sig:"Only Onset (attacks)",prob:"A drum pattern contains hundreds of onsets — far more than section boundaries",verdict:"INSUFFICIENT",c:C.red},
    {sig:"SSM (self-similarity)",prob:"Detects repeated Chorus patterns, but timestamp precision depends on frame resolution",verdict:"POWERFUL — but incomplete alone",c:C.amber},
  ];
  examples.forEach(({sig,prob,verdict,c},i)=>{
    const col=i%2,row=Math.floor(i/2);
    const x=0.3+col*6.55,y=2.38+row*2.2;
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:2.0,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:0.45,fill:{color:C.navyMid},line:{color:C.navyMid}});
    s.addText("📊 "+sig,{x:x+0.1,y,w:5.9,h:0.45,fontSize:12,color:C.white,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText(prob,{x:x+0.1,y:y+0.5,w:5.9,h:1.0,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:x+0.1,y:y+1.58,w:5.9,h:0.32,fill:{color:c},line:{color:c},rounding:0.05});
    s.addText("→ "+verdict,{x:x+0.1,y:y+1.58,w:5.9,h:0.32,fontSize:10,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
  });
  qbar(s,"Conclusion: without combining multiple signals, no system can be reliable across all music genres",6.72);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — WHY DISTRIBUTED ARCHITECTURE?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Why Distributed Architecture?","DSP computation must not block the HTTP request");
  badge(s,5);

  vsBox(s,
    "❌ Monolithic (compute inside the API)",
    ["HTTP request waits for all 4 algorithms to finish","One slow algorithm blocks every user","Adding a new algorithm requires modifying the API","Parallel execution is difficult — thread blocking risk","Scaling means scaling the entire service"],
    "✓ Distributed Workers (our choice)",
    ["API only records the task and publishes a message","Algorithms run independently and in parallel","New algorithm = new worker; API unchanged","Each worker can be scaled independently","One worker crash does not affect the others"],
    0.3,1.42,12.7,4.2,C.red,C.green);

  s.addShape(pres.shapes.RECTANGLE,{x:0.3,y:5.82,w:12.7,h:0.45,fill:{color:C.navyMid},line:{color:C.navyMid}});
  s.addText("Why RabbitMQ specifically?",{x:0.3,y:5.82,w:12.7,h:0.45,fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
  const mq=[
    ["Topic Exchange","Each algorithm bound to its own routing key — one message reaches only its intended worker"],
    ["ACK mechanism","Message stays in queue until the worker processes it; crash causes re-delivery, not loss"],
    ["Service isolation","Backend and workers do not call each other directly — one stopping does not cascade"],
    ["Single result channel","All workers publish to segmentation.result — ResultListener aggregates from one place"],
  ];
  mq.forEach(([t,d],i)=>{
    const col=i%2,row=Math.floor(i/2);
    const x=0.3+col*6.55,y=6.38+row*0.58;
    s.addText("▸ "+t+":",{x,y,w:2.2,h:0.5,fontSize:10,color:C.copperLt,bold:true,fontFace:"Calibri",align:"left",valign:"middle"});
    s.addText(d,{x:x+2.2,y,w:4.2,h:0.5,fontSize:10,color:"CBD5E0",fontFace:"Calibri",align:"left",valign:"middle"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — HOW A REQUEST FLOWS
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"How a Request Flows Through the System","The order of each step reflects a deliberate design decision");
  badge(s,6);

  const steps=[
    {n:"1",t:"Validation first",r:"A malformed parameter must be caught at the API boundary — Pydantic schemas prevent a bad request from reaching a worker.",c:C.copper},
    {n:"2",t:"Task written to DB before publishing",r:"If a worker responds very quickly, the ResultListener must already have a task record to write into. Persistence before dispatch is a guarantee.",c:C.teal},
    {n:"3",t:"Fusion is NOT dispatched immediately",r:"The fusion worker needs 4 base algorithm results as input. Orchestrator adds fusion to the 'expected' list but holds it back from the 'dispatch' list.",c:C.green},
    {n:"4",t:"Workers run in parallel",r:"RabbitMQ delivers a separate message to each algorithm's queue. custom_librosa, Foote, CNMF, and SCluster do not wait for each other.",c:C.amber},
    {n:"5",t:"Results are normalized as they arrive",r:"ResultListener normalizes each incoming result and writes it to DB. Frontend sees partial results in real time via SSE stream.",c:C.purple},
    {n:"6",t:"Fusion starts only after all bases resolve",r:"_maybe_dispatch_fusion() waits for all 4 baselines. Fusing with incomplete information would bias the result toward whichever algorithm finished first.",c:C.navy},
  ];
  steps.forEach(({n,t,r,c},i)=>{
    const col=i%2,row=Math.floor(i/2);
    const x=0.3+col*6.55,y=1.45+row*1.92;
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:1.75,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:x+0.08,y:y+0.1,w:0.48,h:0.48,fill:{color:c},line:{color:c},rounding:0.08});
    s.addText(n,{x:x+0.08,y:y+0.1,w:0.48,h:0.48,fontSize:14,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
    s.addText(t,{x:x+0.65,y:y+0.1,w:5.3,h:0.42,fontSize:12,color:C.navy,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(r,{x:x+0.1,y:y+0.62,w:5.9,h:1.05,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — WHY A COMMON RESULT SCHEMA?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Why a Common Result Schema?","Different algorithms must speak the same language for fusion to work");
  badge(s,7);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:1.42,w:12.7,h:0.78,
    fill:{color:"FFF7ED"},line:{color:C.amber},rounding:0.07});
  s.addText("Without a shared schema: the fusion service would need a separate parser for each algorithm. custom_librosa, Foote, CNMF, and SCluster all produce different raw formats. Adding a new algorithm would break fusion.",{
    x:0.5,y:1.42,w:12.3,h:0.78,fontSize:12,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle"});

  rcCard(s,0.3,2.42,4.0,4.95,
    "Each algorithm produced a different raw format — Foote: boundary timestamps only; CNMF: labeled segment list; custom: feature metadata included",
    "AlgorithmResult schema: task_id, status, worker_type, algorithm, duration, boundaries[], segments[], diagnostics{}",
    "Fusion and evaluation never run algorithm-specific code. Writing a new worker means only conforming to this schema.",C.copper);

  rcCard(s,4.5,2.42,4.0,4.95,
    "MSAF's raw label was literally 'Verse'. That claim is too strong — is it really a Verse, or just a cluster that happens to repeat?",
    "Two layers: structural_label (A/B/C — similarity claim) and semantic_label (Intro/Verse/Chorus — musical role claim, heuristic)",
    "The system does not present what it does not know as fact. Each label carries confidence and semantic_reason fields.",C.teal);

  rcCard(s,8.7,2.42,4.3,4.95,
    "ResultListener must serve two consumers at once: the frontend needs a simple segment list; fusion needs the full normalized result",
    "results[algo] → segment list (frontend). results[algo__result] → full result (fusion). results[algo__diagnostics] → explainability.",
    "Old frontend kept working without changes when the new fusion system was added. Backwards compatibility and extensibility together.",C.green);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — SECTION DIVIDER
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  div(s,"Segmentation Algorithms","Why these 4, and why do they work together?");
  badge(s,8);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — CUSTOM LIBROSA: WHY MULTI-FEATURE?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Custom Librosa Pipeline — Why Multi-Feature?","Each signal sees a different type of transition");
  badge(s,9);

  s.addText("We built the pipeline in this order for a reason:",{x:0.3,y:1.42,w:6,h:0.35,fontSize:12,color:C.navy,bold:true,fontFace:"Georgia",align:"left"});
  flowRow(s,[
    {label:"Load Audio\n(ffmpeg\nmono 22050Hz)",color:C.slate},
    {label:"Detect\nActive Region\n(crop silence)",color:C.teal},
    {label:"Extract\nFeatures\n(Chroma+MFCC)",color:C.green},
    {label:"Build SSM\n(every frame\nvs every frame)",color:C.copper},
    {label:"Generate\nCandidates\n(6 signals)",color:C.amber},
    {label:"Fuse\n(feature fusion\n+ snapping)",color:C.purple},
  ],0.3,1.9,2.0,1.6);

  const whys=[
    {s:"1. Why detect the active region first?",
     w:"Silence distorts SSM calculations and segment duration measurements. We analyze only the musical content, then add active_start back to all timestamps at the end.",c:C.teal},
    {s:"2. Why is SSM the heaviest step?",
     w:"Every frame is compared to every other frame (O(N²)). Repeated Chorus sections appear as bright diagonal blocks in the matrix — structural information no other signal can provide.",c:C.copper},
    {s:"3. Why run 6 signals simultaneously?",
     w:"RMS captures energy change, Chroma captures harmonic shift, Onset captures attack density, Beat captures rhythmic alignment, SSM captures structural repetition, Lyrics capture text boundaries. None sees what the others see.",c:C.green},
    {s:"4. Why is snapping the final step?",
     w:"SSM finds the right region but timestamp precision is limited by frame resolution. At the very end we snap to a nearby strong onset or beat — structural detector answers 'where', onset answers 'exactly when'.",c:C.amber},
  ];
  whys.forEach(({s:st,w,c},i)=>{
    const col=i%2,row=Math.floor(i/2);
    const x=0.3+col*6.55,y=3.75+row*1.65;
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:1.5,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:0.06,h:1.5,fill:{color:c},line:{color:c}});
    s.addText(st,{x:x+0.18,y:y+0.08,w:5.8,h:0.38,fontSize:12,color:c,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(w,{x:x+0.18,y:y+0.5,w:5.8,h:0.95,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — WHY IS SSM THE ANCHOR SIGNAL?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Self-Similarity Matrix — Why the Central Signal?","Repetition information exists only in the SSM");
  badge(s,10);

  s.addText("What we see in the SSM:",{x:0.3,y:1.42,w:4,h:0.35,fontSize:13,color:C.navy,bold:true,fontFace:"Georgia"});

  // Simplified SSM grid
  const gx=0.3,gy=1.88,cs=0.46,gs=8;
  const sec=(i)=>i<1?0:i<3?1:i<5?2:i<7?1:2;
  const clr=(r,c)=>sec(r)===sec(c)?"2D6A4F":r===c?"1B2A4A":"E2E8F0";
  for(let r=0;r<gs;r++) for(let c=0;c<gs;c++)
    s.addShape(pres.shapes.RECTANGLE,{x:gx+c*cs,y:gy+r*cs,w:cs-0.02,h:cs-0.02,fill:{color:clr(r,c)},line:{color:"FFFFFF"}});
  ["Intro","Verse","Verse","Chorus","Chorus","Verse","Verse","Chorus"].forEach((l,i)=>{
    s.addText(l,{x:gx+i*cs+0.01,y:gy+gs*cs+0.04,w:cs,h:0.3,fontSize:6.5,color:C.slate,align:"center"});
    s.addText(l,{x:gx-0.55,y:gy+i*cs,w:0.54,h:cs,fontSize:6.5,color:C.slate,align:"right",valign:"middle"});
  });
  s.addText("🟩 = similar regions (Verse–Verse, Chorus–Chorus)",{
    x:gx,y:gy+gs*cs+0.42,w:gs*cs,h:0.35,fontSize:9,color:C.green,fontFace:"Calibri",align:"center"});

  const pts=[
    {t:"Instantaneous change vs structural repetition",
     b:"RMS and onset measure what is happening right now. SSM asks 'do two distant moments in this track sound similar?' Only SSM can detect that the Chorus is appearing for the third time.",c:C.copper},
    {t:"Why the highest weight (0.42)?",
     b:"In feature-level fusion, a strong SSM candidate (confidence ≥ 0.5) can pass acceptance on its own. Other signals provide support — without SSM, real section boundaries are likely to be missed.",c:C.green},
    {t:"Why transposition invariance?",
     b:"The same motif can repeat in a different key. We try all 12 pitch shifts and keep the maximum similarity. Structural repetition survives even when the key changes.",c:C.teal},
    {t:"Why diagonal smoothing and enhancement?",
     b:"The raw SSM is noisy. We smooth along diagonals — accounting for tempo variation — to strengthen repetition paths. The checkerboard kernel produces its highest response exactly at boundaries.",c:C.amber},
  ];
  pts.forEach(({t,b,c},i)=>{
    const y=1.42+i*1.55;
    s.addShape(pres.shapes.RECTANGLE,{x:4.4,y,w:8.7,h:1.42,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE,{x:4.4,y,w:0.06,h:1.42,fill:{color:c},line:{color:c}});
    s.addText(t,{x:4.58,y:y+0.08,w:8.4,h:0.38,fontSize:12,color:c,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(b,{x:4.58,y:y+0.5,w:8.4,h:0.88,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — MSAF BASELINES: WHY 3 DIFFERENT METHODS?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"MSAF Baselines — Why 3 Different Methods?","Different mathematical assumptions produce different error profiles");
  badge(s,11);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"EFF6FF"},line:{color:C.teal},rounding:0.07});
  s.addText("The value of fusion is not combining 4 variations of the same algorithm — it is combining independent opinions from different mathematical perspectives. Similar error profiles make fusion pointless.",{
    x:0.5,y:1.42,w:12.3,h:0.72,fontSize:12,color:C.navy,fontFace:"Calibri",align:"left",valign:"middle"});

  const algos=[
    {name:"Foote",math:"Local Novelty Detection",
     persp:"'How much has the structure changed compared to a moment ago?'",
     str:"Captures sharp, local transitions. Fast and explainable.",
     wk:"Sensitive to non-structural transients (e.g., drum fills).",
     wt:"0.15 — local signal only, no global structural knowledge",c:"5B21B6"},
    {name:"CNMF",math:"Convex Non-negative Matrix Factorization",
     persp:"'What hidden recurring components exist in this track?'",
     str:"Factorizes latent patterns — extracts structure already inside the data.",
     wk:"Factorization rank (k) choice affects the number of boundaries.",
     wt:"0.20 — less sensitive to local noise than Foote",c:C.teal},
    {name:"SCluster",math:"Spectral Clustering on Affinity Graph",
     persp:"'How does the entire track divide into global clusters?'",
     str:"Uses global similarity structure, not local change. Captures repeated sections as clusters.",
     wk:"Cluster count parameter determines segment granularity.",
     wt:"0.30 — global structure; second-highest weight after custom",c:C.green},
  ];
  algos.forEach(({name,math,persp,str,wk,wt,c},i)=>{
    const x=0.3+i*4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:2.35,w:4.05,h:5.25,
      fill:{color:C.white},line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.07},rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE,{x,y:2.35,w:4.05,h:0.55,fill:{color:c},line:{color:c}});
    s.addText(name,{x,y:2.35,w:4.05,h:0.55,fontSize:20,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle",margin:0});
    s.addText(math,{x:x+0.12,y:2.97,w:3.81,h:0.38,fontSize:10,color:c,bold:true,fontFace:"Calibri",align:"center"});
    s.addText("Perspective",{x:x+0.12,y:3.4,w:3.81,h:0.28,fontSize:9,color:C.gray,bold:true,fontFace:"Calibri"});
    s.addText(persp,{x:x+0.12,y:3.68,w:3.81,h:0.55,fontSize:11,color:C.navy,fontFace:"Calibri",italic:true});
    s.addShape(pres.shapes.RECTANGLE,{x:x+0.12,y:4.28,w:3.81,h:0.01,fill:{color:C.grayLt},line:{color:C.grayLt}});
    s.addText("✓ "+str,{x:x+0.12,y:4.32,w:3.81,h:0.85,fontSize:10,color:C.green,fontFace:"Calibri",valign:"top"});
    s.addText("⚠ "+wk,{x:x+0.12,y:5.22,w:3.81,h:0.75,fontSize:10,color:C.amber,fontFace:"Calibri",valign:"top"});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:x+0.12,y:6.02,w:3.81,h:0.38,fill:{color:c},line:{color:c},rounding:0.05});
    s.addText("Weight: "+wt,{x:x+0.12,y:6.02,w:3.81,h:0.38,fontSize:9,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — FEATURE-LEVEL FUSION: WHY THIS FORMULA?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Feature-Level Fusion — Why This Scoring Formula?","We reward agreement between independent sources");
  badge(s,12);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:1.42,w:12.7,h:0.68,
    fill:{color:"FFF8F0"},line:{color:C.amber},rounding:0.07});
  s.addText("Challenge: 6 signals each produce multiple candidate boundaries, offset by tens of milliseconds from each other. Which one is real?  —  Confidence grows with the number of independent sources agreeing.",{
    x:0.5,y:1.42,w:12.3,h:0.68,fontSize:12,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle"});

  const decisions=[
    {d:"Why temporal grouping?",
     b:"SSM says 30.8 s, RMS says 31.5 s — both detected the same transition, just in different frames. Candidates within 2.75 s are treated as one group.",c:C.copper},
    {d:"Why only one vote per source?",
     b:"A noisy SSM can produce 10 peaks in the same region. Only the highest-confidence candidate from each source counts. This prevents a single feature from dominating the vote.",c:C.teal},
    {d:"Why add an agreement bonus?",
     b:"A lone SSM vote gives score = 0.42 × conf. But if 3 independent signals point to the same region, we add a bonus (up to 0.15). Consensus earns extra trust.",c:C.green},
    {d:"Why can SSM pass acceptance alone?",
     b:"SSM is the only signal carrying structural repetition information. Even if other signals are weak, a strong SSM candidate (conf ≥ 0.5) is kept. Without this rule, real section boundaries would be missed.",c:C.amber},
  ];
  decisions.forEach(({d,b,c},i)=>{
    const col=i%2,row=Math.floor(i/2);
    const x=0.3+col*6.55,y=2.3+row*2.35;
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:2.2,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:0.07,h:2.2,fill:{color:c},line:{color:c}});
    s.addText(d,{x:x+0.2,y:y+0.1,w:5.8,h:0.45,fontSize:13,color:c,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(b,{x:x+0.2,y:y+0.6,w:5.8,h:1.5,fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });

  s.addShape(pres.shapes.RECTANGLE,{x:0.3,y:7.05,w:12.7,h:0.42,fill:{color:C.navyMid},line:{color:C.navyMid}});
  s.addText("score = Σ(source_weight × candidate_confidence)  +  min(0.15, 0.035 × (source_count − 1))     accept: score ≥ 0.30  OR  strong SSM",{
    x:0.5,y:7.05,w:12.3,h:0.42,fontSize:10.5,color:C.copperLt,bold:true,fontFace:"Calibri",align:"center",valign:"middle"});
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — WHY TWO FUSION LEVELS?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Why Two Levels of Fusion?","Feature fusion is not enough — why algorithm-level fusion is necessary");
  badge(s,13);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"F0FDF4"},line:{color:C.green},rounding:0.07});
  s.addText("Feature-level fusion lives inside a single algorithm. custom_librosa's output already combines 6 signals. But custom_librosa itself can still make mistakes. The second fusion level balances the errors of different algorithms.",{
    x:0.5,y:1.42,w:12.3,h:0.72,fontSize:12,color:C.green,fontFace:"Calibri",align:"left",valign:"middle"});

  s.addText("Level 1 — Feature Fusion (inside custom_librosa):",{x:0.3,y:2.32,w:12.7,h:0.35,fontSize:13,color:C.navy,bold:true,fontFace:"Georgia"});
  flowRow(s,[
    {label:"RMS\nsignal",color:C.copper},{label:"Chroma\nsignal",color:C.teal},
    {label:"Onset\nsignal",color:C.green},{label:"SSM\nsignal",color:C.amber},{label:"Beat\nsignal",color:C.purple},
  ],0.5,2.73,1.8,0.85);
  s.addText("→",{x:10.2,y:2.88,w:0.3,h:0.5,fontSize:18,color:C.copper,bold:true,align:"center"});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:10.55,y:2.73,w:2.5,h:0.85,fill:{color:C.copper},line:{color:C.copper},rounding:0.08});
  s.addText("custom_librosa\nresult",{x:10.55,y:2.73,w:2.5,h:0.85,fontSize:11,color:C.white,bold:true,align:"center",valign:"middle"});

  s.addShape(pres.shapes.RECTANGLE,{x:0.3,y:3.73,w:12.7,h:0.01,fill:{color:C.grayLt},line:{color:C.grayLt}});
  s.addText("Level 2 — Algorithm Fusion (separate worker):",{x:0.3,y:3.82,w:12.7,h:0.35,fontSize:13,color:C.navy,bold:true,fontFace:"Georgia"});
  flowRow(s,[
    {label:"custom_librosa\nresult",color:C.copper},{label:"Foote\nresult",color:"5B21B6"},
    {label:"CNMF\nresult",color:C.teal},{label:"SCluster\nresult",color:C.green},
  ],0.5,4.23,2.3,0.85);
  s.addText("→",{x:10.2,y:4.38,w:0.3,h:0.5,fontSize:18,color:C.copper,bold:true,align:"center"});
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:10.55,y:4.23,w:2.5,h:0.85,fill:{color:C.navy},line:{color:C.navy},rounding:0.08});
  s.addText("FUSION\nresult",{x:10.55,y:4.23,w:2.5,h:0.85,fontSize:11,color:C.white,bold:true,align:"center",valign:"middle"});

  const tbl=[
    ["","Feature-Level Fusion","Algorithm-Level Fusion"],
    ["Voters","Signal sources (SSM, RMS, onset…)","Independent segmentation algorithms"],
    ["Input","Raw audio feature candidates","Completed AlgorithmResult objects"],
    ["Location","Inside custom_librosa pipeline","Separate FusionWorker service"],
    ["Goal","Signal diversity within one pipeline","Balancing different error profiles"],
  ];
  s.addTable(tbl.map((row,ri)=>row.map((cell,ci)=>({
    text:cell,
    options:{fontSize:ri===0||ci===0?11:10.5,bold:ri===0||ci===0,
      color:ri===0?C.white:ci===0?C.navy:C.slate,
      fill:ri===0?{color:C.navy}:ci===0?{color:"EEF2FF"}:{color:C.white},align:"center"},
  }))),{x:0.3,y:5.28,w:12.7,h:2.05,colW:[2.5,5.1,5.1],border:{pt:1,color:C.grayLt}});
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — SECTION DIVIDER
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  div(s,"Algorithm-Level Fusion","Why weighted voting? Why wait for all results?");
  badge(s,14);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — WHY WEIGHTED VOTING?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Why Weighted Voting? — Simple Majority Is Not Enough","Algorithms are not equally reliable; confidence matters too");
  badge(s,15);

  const alts=[
    {t:"❌ Simple Majority Vote",items:[
      "Treats every algorithm as equally reliable",
      "Ignores the confidence value of each boundary",
      "2/4 votes accepted — weak evidence",
      "An over-segmenting algorithm can win by producing more boundaries",
    ],c:C.red},
    {t:"❌ Simple Average (mean timestamp)",items:[
      "Does not group votes that target the same boundary",
      "The midpoint of two boundaries 10 s apart is meaningless",
      "Ignores confidence and algorithm weights",
    ],c:C.amber},
    {t:"✓ Weighted Voting (our choice)",items:[
      "score = Σ(algorithm_weight × boundary_confidence)",
      "custom_librosa (0.35) > scluster (0.30) > cnmf (0.20) > foote (0.15)",
      "Each algorithm casts at most 1 vote per boundary group",
      "Accept: score ≥ threshold  OR  source_count ≥ required_votes",
    ],c:C.green},
  ];
  alts.forEach(({t,items,c},i)=>{
    const x=0.3+i*4.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:1.42,w:4.05,h:4.5,
      fill:{color:C.white},line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.08},rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE,{x,y:1.42,w:4.05,h:0.58,fill:{color:c},line:{color:c}});
    s.addText(t,{x:x+0.1,y:1.42,w:3.85,h:0.58,fontSize:12,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
    bullets(s,items,x+0.1,2.05,3.85,3.8,11,C.slate);
  });

  s.addShape(pres.shapes.RECTANGLE,{x:0.3,y:6.1,w:12.7,h:0.42,fill:{color:C.navyMid},line:{color:C.navyMid}});
  s.addText("Why two acceptance conditions (score OR vote count)?",{x:0.3,y:6.1,w:12.7,h:0.42,fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
  [
    ["Score threshold alone","If confidence calibration differs across algorithms, a high-weight algorithm could dominate blindly"],
    ["Vote count fallback","If two independent algorithms agree on a region — even with low confidence — consensus is preserved"],
  ].forEach(([t,d],i)=>{
    s.addText("▸ "+t+": "+d,{x:0.4,y:6.6+i*0.35,w:12.5,h:0.3,fontSize:10.5,color:"CBD5E0",fontFace:"Calibri",align:"left",valign:"middle"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — WHY WAIT FOR ALL RESULTS?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Why Does Fusion Wait for All Base Results?","Fusing with incomplete information biases the outcome");
  badge(s,16);

  const reasons=[
    {t:"Early fusion produces biased results",
     b:"If Foote finishes fast and custom_librosa is still running: fusion runs on Foote data only. The result appears as if 4 algorithms agreed — but only 1 actually spoke. This false confidence is dangerous.",c:C.red},
    {t:"All perspectives together make consensus meaningful",
     b:"custom_librosa brings structural SSM, SCluster brings global clustering, CNMF brings latent patterns, Foote brings local novelty. When all four independently point to the same boundary, confidence is genuinely earned.",c:C.green},
    {t:"Minimum 2 successful results required",
     b:"Once all 4 baselines are resolved, if fewer than 2 produced successful results, fusion is pointless. ResultListener directly generates a 'failed fusion' result — no unnecessary computation.",c:C.teal},
    {t:"Known limitation: what if a worker disappears?",
     b:"If a worker throws an exception, BaseWorker publishes a 'failed' result → counted as resolved. But if a worker crashes without publishing anything, the task stays in 'processing' forever. Watchdog/timeout is future work.",c:C.amber},
  ];
  reasons.forEach(({t,b,c},i)=>{
    const col=i%2,row=Math.floor(i/2);
    const x=0.3+col*6.55,y=1.42+row*2.88;
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:2.68,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:0.07,h:2.68,fill:{color:c},line:{color:c}});
    s.addText(t,{x:x+0.2,y:y+0.1,w:5.8,h:0.55,fontSize:13,color:c,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText(b,{x:x+0.2,y:y+0.72,w:5.8,h:1.9,fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });
  qbar(s,"Completeness is required for trust: we collect consensus from all 4 perspectives — or we do not fuse at all",7.24);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 17 — WHY TWO-LAYER LABELS?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Why Two-Layer Labeling?","A similarity claim and a musical role claim require different levels of evidence");
  badge(s,17);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"FFF8F0"},line:{color:C.amber},rounding:0.07});
  s.addText("'These two segments sound similar to each other' and 'this segment is the Chorus' are not the same strength of claim. The first is an audio descriptor measurement. The second is a musical interpretation.",{
    x:0.5,y:1.42,w:12.3,h:0.72,fontSize:13,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle"});

  // Label example
  [{sl:"A",sem:"Intro",c:C.copper},{sl:"B",sem:"Verse",c:C.teal},{sl:"C",sem:"Chorus",c:C.green},
   {sl:"B",sem:"Verse",c:C.teal},{sl:"C",sem:"Chorus",c:C.green},{sl:"D",sem:"Bridge",c:C.amber},{sl:"A",sem:"Outro",c:C.copper}].forEach(({sl,sem,c},i,arr)=>{
    const w=13.3/arr.length;
    s.addShape(pres.shapes.RECTANGLE,{x:i*w,y:2.32,w:w-0.03,h:0.6,fill:{color:c},line:{color:c}});
    s.addText(`Struct: ${sl}`,{x:i*w,y:2.32,w:w-0.03,h:0.32,fontSize:10,color:C.white,bold:true,align:"center",valign:"middle"});
    s.addText(sem,{x:i*w,y:2.64,w:w-0.03,h:0.28,fontSize:9,color:C.white,align:"center",valign:"middle"});
  });

  [
    {t:"Structural Label (A/B/C)",sub:"'This segment sounds like that one' — measurable, reliable",
     pts:["Clustered from Chroma, MFCC, RMS, onset density descriptors",
       "Agglomerative clustering; best k chosen by silhouette score",
       "A = most frequent cluster, B = second — does NOT mean Verse",
       "Honest claim: audio descriptor similarity was measured"],c:C.copper},
    {t:"Semantic Label (Intro/Verse/Chorus…)",sub:"'This section plays the musical role of X' — heuristic, weaker",
     pts:["Heuristic rules: unique first segment in early 20% → Intro",
       "Repeated cluster with higher RMS energy → Chorus candidate",
       "Does NOT overwrite the structural label — stored separately",
       "Each label carries semantic_confidence and semantic_reason",
       "If evidence is insufficient: 'Unknown' or 'Early/Middle/Late'"],c:C.teal},
  ].forEach(({t,sub,pts,c},i)=>{
    const x=0.3+i*6.55;
    s.addShape(pres.shapes.RECTANGLE,{x,y:3.12,w:6.1,h:4.28,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.RECTANGLE,{x,y:3.12,w:6.1,h:0.55,fill:{color:c},line:{color:c}});
    s.addText(t,{x:x+0.1,y:3.12,w:5.9,h:0.55,fontSize:14,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
    s.addText(sub,{x:x+0.1,y:3.72,w:5.9,h:0.45,fontSize:11,color:c,bold:true,fontFace:"Calibri",italic:true,align:"left",valign:"middle"});
    bullets(s,pts,x+0.1,4.22,5.9,3.1,11,C.slate);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 18 — EVALUATION: WHY TWO TOLERANCES?
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Evaluation — Why Two Tolerances?","One metric can hide which specific problem the system has");
  badge(s,18);

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:1.42,w:12.7,h:0.72,
    fill:{color:"EFF6FF"},line:{color:C.teal},rounding:0.07});
  s.addText("Evaluating with a single tolerance can obscure a system's real weakness. Without two tolerances, you cannot distinguish a system that finds the right region (but is imprecise in timing) from one that misses boundaries entirely.",{
    x:0.5,y:1.42,w:12.3,h:0.72,fontSize:12,color:C.navy,fontFace:"Calibri",align:"left",valign:"middle"});

  [
    {tol:"±0.5 s — Strict",q:"Did we find the exact moment?",
     m:"Measures timestamp localization precision. Low F1@0.5 means the algorithm finds the right structural region but is imprecise about the exact second.",
     low:"Snapping, frame resolution, or SSM smoothing is shifting the timestamp",c:C.copper},
    {tol:"±3.0 s — Lenient",q:"Did we detect the right region?",
     m:"Measures structural region detection. If F1@3.0 is high, the system understands the musical structure. High @3 but low @0.5 → good region detection, weak timing.",
     low:"Fundamental boundary detection or boundary count problem",c:C.teal},
  ].forEach(({tol,q,m,low,c},i)=>{
    const x=0.3+i*6.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y:2.32,w:6.1,h:3.65,
      fill:{color:C.white},line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:5,offset:1,angle:135,opacity:0.07},rounding:0.08});
    s.addShape(pres.shapes.RECTANGLE,{x,y:2.32,w:6.1,h:0.65,fill:{color:c},line:{color:c}});
    s.addText(tol,{x:x+0.1,y:2.32,w:5.9,h:0.65,fontSize:16,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
    s.addText("Q: "+q,{x:x+0.1,y:3.03,w:5.9,h:0.38,fontSize:13,color:c,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText(m,{x:x+0.1,y:3.47,w:5.9,h:1.05,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:x+0.1,y:4.6,w:5.9,h:1.22,fill:{color:"FFF7ED"},line:{color:C.amber},rounding:0.05});
    s.addText("If low: "+low,{x:x+0.2,y:4.65,w:5.7,h:1.12,fontSize:11,color:C.amber,fontFace:"Calibri",align:"left",valign:"middle"});
  });

  s.addShape(pres.shapes.RECTANGLE,{x:0.3,y:6.1,w:12.7,h:0.42,fill:{color:C.navyMid},line:{color:C.navyMid}});
  s.addText("Interpretation guide",{x:0.3,y:6.1,w:12.7,h:0.42,fontSize:12,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});
  [
    ["F1@3 high,  F1@0.5 low","Correct region detected — timestamp precision is the bottleneck → improve snapping"],
    ["Both high","Both region and timing are correct → well-functioning system"],
    ["Both low","Fundamental boundary detection issue → wrong boundary count or location"],
  ].forEach(([sc,mn],i)=>{
    const y=6.6+i*0.38;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y,w:4.5,h:0.32,fill:{color:C.copper},line:{color:C.copper},rounding:0.05});
    s.addText(sc,{x:0.3,y,w:4.5,h:0.32,fontSize:9.5,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
    s.addText("→  "+mn,{x:4.9,y,w:8.2,h:0.32,fontSize:10.5,color:C.grayLt,fontFace:"Calibri",align:"left",valign:"middle"});
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 19 — FULL DATASET RESULTS
// Per-track F1 comparison: custom_librosa vs baseline at ±0.5 s (actual data)
// Summary table includes 3.0 s column (TBD — to be run before presentation)
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Full Dataset Evaluation Results","14 SALAMI tracks · custom_librosa vs MSAF Baseline · mir_eval.segment.detection");
  badge(s,19);

  // ── Per-track data (0.5 s tolerance, actual) ──────────────────────────────
  const tracks=[
    {id:"957", name:"Tip My Glass",       cl:0.000, bl:0.000},
    {id:"959", name:"STK",                cl:0.167, bl:0.222},
    {id:"965", name:"The Unwanted Brother",cl:0.056,bl:0.100},
    {id:"967", name:"Europa Baby",         cl:0.059, bl:0.091},
    {id:"971", name:"Figure It Out",      cl:0.250, bl:0.087},
    {id:"973", name:"Blue Veins",         cl:0.129, bl:0.191},
    {id:"992", name:"Mexican Radio",      cl:0.774, bl:0.071},
    {id:"1006",name:"The Navy Song",      cl:0.214, bl:0.000},
    {id:"1013",name:"Those Get Me Out",   cl:0.059, bl:0.000},
    {id:"1032",name:"CBC Demo 1",         cl:0.700, bl:0.111},
    {id:"1038",name:"ShimSham Honey",     cl:0.074, bl:0.074},
    {id:"1046",name:"People",             cl:0.074, bl:0.083},
    {id:"1048",name:"7 Escape Artist",    cl:0.105, bl:0.000},
    {id:"1050",name:"Going Up Country",   cl:0.286, bl:0.100},
  ];

  const avgCL = tracks.reduce((a,t)=>a+t.cl,0)/tracks.length;
  const avgBL = tracks.reduce((a,t)=>a+t.bl,0)/tracks.length;

  // Chart area
  const chartX=0.3, chartY=1.45, chartW=8.0, chartH=5.55;
  const barH=0.28, gap=0.12, barGroup=barH*2+gap+0.04;

  s.addShape(pres.shapes.RECTANGLE,{x:chartX,y:chartY,w:chartW,h:chartH,fill:{color:C.white},line:{color:C.grayLt}});

  // Avg line
  [avgCL,avgBL].forEach((avg,ai)=>{
    const lx=chartX+avg*chartW;
    s.addShape(pres.shapes.RECTANGLE,{
      x:lx,y:chartY,w:0.02,h:chartH,
      fill:{color:ai===0?C.copper:"4A5568"},line:{color:ai===0?C.copper:"4A5568"},
    });
  });

  tracks.forEach(({name,cl,bl},i)=>{
    const gy=chartY+0.15+i*(barGroup);
    // CL bar
    s.addShape(pres.shapes.RECTANGLE,{x:chartX,y:gy,w:cl*chartW,h:barH,fill:{color:C.copper},line:{color:C.copper}});
    // BL bar
    s.addShape(pres.shapes.RECTANGLE,{x:chartX,y:gy+barH+0.04,w:bl*chartW,h:barH,fill:{color:"9CA3AF"},line:{color:"9CA3AF"}});
    // F1 labels
    if(cl>0.02) s.addText(cl.toFixed(3),{x:chartX+cl*chartW+0.03,y:gy,w:0.55,h:barH,fontSize:7,color:C.copper,fontFace:"Calibri",align:"left",valign:"middle"});
    if(bl>0.02) s.addText(bl.toFixed(3),{x:chartX+bl*chartW+0.03,y:gy+barH+0.04,w:0.55,h:barH,fontSize:7,color:"6B7280",fontFace:"Calibri",align:"left",valign:"middle"});
  });

  // Track labels on left side — outside chart
  tracks.forEach(({name},i)=>{
    const gy=chartY+0.15+i*(barGroup);
    s.addText(name,{x:chartX-0.05,y:gy,w:0,h:barGroup,fontSize:0,color:C.white}); // spacer
  });

  // Legend
  s.addShape(pres.shapes.RECTANGLE,{x:chartX,y:chartY+chartH+0.12,w:0.22,h:0.18,fill:{color:C.copper},line:{color:C.copper}});
  s.addText("custom_librosa  (avg F1 = "+avgCL.toFixed(3)+")",{x:chartX+0.28,y:chartY+chartH+0.08,w:4,h:0.28,fontSize:10,color:C.copper,fontFace:"Calibri",valign:"middle"});
  s.addShape(pres.shapes.RECTANGLE,{x:chartX+4.5,y:chartY+chartH+0.12,w:0.22,h:0.18,fill:{color:"9CA3AF"},line:{color:"9CA3AF"}});
  s.addText("MSAF Baseline  (avg F1 = "+avgBL.toFixed(3)+")",{x:chartX+4.78,y:chartY+chartH+0.08,w:4,h:0.28,fontSize:10,color:"6B7280",fontFace:"Calibri",valign:"middle"});

  // Right panel: summary table + 3s column
  s.addShape(pres.shapes.RECTANGLE,{x:8.6,y:chartY,w:4.5,h:chartH,fill:{color:C.white},line:{color:C.grayLt}});
  s.addShape(pres.shapes.RECTANGLE,{x:8.6,y:chartY,w:4.5,h:0.42,fill:{color:C.navy},line:{color:C.navy}});
  s.addText("Summary",{x:8.6,y:chartY,w:4.5,h:0.42,fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"center",valign:"middle"});

  // ±0.5 s section
  s.addShape(pres.shapes.RECTANGLE,{x:8.6,y:chartY+0.48,w:4.5,h:0.3,fill:{color:"F1F5F9"},line:{color:C.grayLt}});
  s.addText("Tolerance ±0.5 s  (actual data)",{x:8.65,y:chartY+0.48,w:4.4,h:0.3,fontSize:9,color:C.navy,bold:true,fontFace:"Calibri",align:"left",valign:"middle"});

  const rows05=[
    ["","Precision","Recall","F1"],
    ["custom_librosa","0.214","0.236","0.210"],
    ["MSAF Baseline","0.081","0.090","0.081"],
  ];
  s.addTable(rows05.map((row,ri)=>row.map((cell,ci)=>({
    text:cell,
    options:{fontSize:ri===0?9:10,bold:ri===0||ci===0,
      color:ri===0?C.white:ci===0?C.navy:ri===1?C.copper:"6B7280",
      fill:ri===0?{color:C.navyMid}:ci===0?{color:"F8FAFC"}:{color:C.white},align:"center"},
  }))),{x:8.6,y:chartY+0.82,w:4.5,h:1.35,colW:[1.8,0.9,0.9,0.9],border:{pt:1,color:C.grayLt}});

  // ±3.0 s section
  s.addShape(pres.shapes.RECTANGLE,{x:8.6,y:chartY+2.28,w:4.5,h:0.3,fill:{color:"FFF7ED"},line:{color:C.amber}});
  s.addText("Tolerance ±3.0 s  — run pending",{x:8.65,y:chartY+2.28,w:4.4,h:0.3,fontSize:9,color:C.amber,bold:true,fontFace:"Calibri",align:"left",valign:"middle"});

  const rows30=[
    ["","Precision","Recall","F1"],
    ["custom_librosa","—","—","—"],
    ["Fusion","—","—","—"],
    ["MSAF Baseline","—","—","—"],
  ];
  s.addTable(rows30.map((row,ri)=>row.map((cell,ci)=>({
    text:cell,
    options:{fontSize:ri===0?9:10,bold:ri===0||ci===0,
      color:ri===0?C.white:ci===0?C.navy:C.amber,
      fill:ri===0?{color:C.navyMid}:ci===0?{color:"F8FAFC"}:{color:"FFFBF0"},align:"center"},
  }))),{x:8.6,y:chartY+2.62,w:4.5,h:1.7,colW:[1.8,0.9,0.9,0.9],border:{pt:1,color:C.grayLt}});

  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:8.6,y:chartY+4.42,w:4.5,h:1.02,
    fill:{color:"FFF7ED"},line:{color:C.amber},rounding:0.05});
  s.addText("⚠ Run batch evaluation with tolerance=3.0 before the presentation to fill in the 3.0 s column. Command:\nPOST /evaluation/batch\n{\"tolerances\":[0.5,3.0]}",{
    x:8.68,y:chartY+4.48,w:4.35,h:0.92,fontSize:9,color:C.amber,fontFace:"Calibri",align:"left",valign:"top"});

  // Note at bottom
  s.addShape(pres.shapes.RECTANGLE,{x:0.3,y:7.08,w:12.7,h:0.38,fill:{color:"F1F5F9"},line:{color:C.grayLt}});
  s.addText("14 evaluable tracks out of ~50 attempted — remaining tracks had download failures in this run. evaluation metric: mir_eval.segment.detection(trim=True).",{
    x:0.5,y:7.08,w:12.3,h:0.38,fontSize:9.5,color:C.gray,fontFace:"Calibri",align:"center",valign:"middle"});
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 20 — DEMO
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Live Demo","The system explains its own decisions through diagnostics");
  badge(s,20);

  const steps=[
    {n:"1",t:"Select a track, launch 4 base algorithms + fusion",
     w:"When the request is sent, the fusion worker does NOT start. 4 independent workers are dispatched in parallel via RabbitMQ with separate routing keys.",c:C.copper},
    {n:"2",t:"Watch the status while partial results arrive",
     w:"Task is still 'processing'. custom_librosa and Foote finished but CNMF and SCluster are still running. Frontend updates in real time via SSE stream.",c:C.teal},
    {n:"3",t:"Compare the different boundary sets from each algorithm",
     w:"This is live evidence for why fusion is needed: you can see 4 algorithms producing different boundary timelines for the same track.",c:C.green},
    {n:"4",t:"Open fusion diagnostics",
     w:"Each fused boundary shows: which algorithms voted, what the raw timestamps were, the weighted score, and whether it was accepted or rejected — fully explainable.",c:C.amber},
    {n:"5",t:"Show evaluation results",
     w:"F1@0.5 and F1@3.0. Read them together: is the region correct (lenient), is the timing correct (strict), or is there a fundamental detection problem?",c:C.purple},
  ];
  steps.forEach(({n,t,w,c},i)=>{
    const y=1.42+i*1.12;
    s.addShape(pres.shapes.RECTANGLE,{x:0.3,y,w:12.7,h:1.0,fill:{color:C.white},line:{color:C.grayLt}});
    s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.35,y:y+0.1,w:0.55,h:0.55,fill:{color:c},line:{color:c},rounding:0.08});
    s.addText(n,{x:0.35,y:y+0.1,w:0.55,h:0.55,fontSize:15,color:C.white,bold:true,align:"center",valign:"middle",margin:0});
    s.addText(t,{x:1.05,y:y+0.08,w:5.5,h:0.4,fontSize:13,color:C.navy,bold:true,fontFace:"Georgia",align:"left",valign:"middle",margin:0});
    s.addText("What to say: "+w,{x:1.05,y:y+0.52,w:11.8,h:0.42,fontSize:11,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x:0.3,y:7.05,w:12.7,h:0.4,fill:{color:"FFF7ED"},line:{color:C.amber},rounding:0.05});
  s.addText("Backup plan: have a previously completed task ID and a screenshot of fusion diagnostics JSON ready. If a worker is slow, use it to demonstrate the async architecture advantage.",{
    x:0.5,y:7.05,w:12.3,h:0.4,fontSize:10,color:C.amber,bold:true,fontFace:"Calibri",align:"center",valign:"middle"});
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 21 — CONCLUSION
// ════════════════════════════════════════════════════════════════════════════
{
  const s=pres.addSlide();
  hdr(s,"Conclusion — What We Built and Why","Every decision has a problem context behind it");
  badge(s,21);

  const decisions=[
    {d:"Distributed architecture",why:"DSP computation must not block the API; each algorithm must be independently scalable",c:C.teal},
    {d:"Common result schema",why:"Fusion must never need to know which algorithm it is reading; adding a new worker means only conforming to the schema",c:C.copper},
    {d:"Feature-level fusion",why:"No single signal is reliable for all genres; consensus from 6 independent sources is more robust",c:C.green},
    {d:"Algorithm-level fusion",why:"Even the best pipeline can make mistakes; error profiles of different mathematical approaches complement each other",c:C.amber},
    {d:"Structural / semantic label separation",why:"A similarity claim is measurable; a musical role claim is heuristic — conflating them would misrepresent scientific honesty",c:C.purple},
    {d:"Two tolerances (0.5 s / 3.0 s)",why:"Finding the right region and finding the exact second are different skills — a single metric hides which one is failing",c:C.navy},
  ];
  decisions.forEach(({d,why,c},i)=>{
    const col=i%2,row=Math.floor(i/2);
    const x=0.3+col*6.55,y=1.42+row*1.82;
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:1.65,fill:{color:C.white},line:{color:C.grayLt},
      shadow:{type:"outer",color:"000000",blur:4,offset:1,angle:135,opacity:0.07}});
    s.addShape(pres.shapes.RECTANGLE,{x,y,w:6.1,h:0.5,fill:{color:c},line:{color:c}});
    s.addText("✓ "+d,{x:x+0.1,y,w:5.9,h:0.5,fontSize:13,color:C.white,bold:true,fontFace:"Georgia",align:"left",valign:"middle"});
    s.addText("Why: "+why,{x:x+0.1,y:y+0.56,w:5.9,h:1.05,fontSize:12,color:C.slate,fontFace:"Calibri",align:"left",valign:"top"});
  });

  s.addShape(pres.shapes.RECTANGLE,{x:0,y:7.38,w:13.3,h:0.42,fill:{color:C.navy},line:{color:C.navy}});
  s.addText("Our contribution is not a new algorithm — it is integrating multi-signal, multi-algorithm, and two-level fusion decisions into a distributed, explainable, and measurable system.",{
    x:0.3,y:7.38,w:12.7,h:0.42,fontSize:10.5,color:C.copperLt,fontFace:"Calibri",align:"center",valign:"middle"});
}

// ─── GENERATE ─────────────────────────────────────────────────────────────────
pres.writeFile({fileName:"docs/MusicSegmentation_Presentation_v2.pptx"})
  .then(()=>console.log("✅  docs/MusicSegmentation_Presentation_v2.pptx created"))
  .catch(e=>{console.error("❌ Error:",e);process.exit(1);});
