
(() => {
window.__res = {}; window.__log = []; window.__done = false; window.__n = 0;
const S = [["Biel", "X201", 47.131, 7.241], ["Biel", "X202", 47.131, 7.236], ["Biel", "X203", 47.137, 7.247], ["Biel", "X204", 47.141, 7.253], ["Biel", "X205", 47.145, 7.265], ["Biel", "X206", 47.139, 7.295], ["Biel", "X207", 47.179, 7.415], ["Biel", "X208", 47.13, 7.262], ["Biel", "X209", 47.135, 7.256], ["Biel", "X210", 47.135, 7.249], ["Biel", "X211", 47.128, 7.249], ["Biel", "X212", 47.134, 7.231], ["Biel", "X213", 47.12, 7.254], ["Biel", "X214", 47.128, 7.266], ["Biel", "X215", 47.142, 7.246], ["Biel", "X216", 47.125, 7.239], ["Biel", "X217", 47.144, 7.284], ["Biel", "X218", 47.133, 7.243], ["Biel", "X219", 47.139, 7.246], ["Biel", "X220", 47.137, 7.251], ["Biel", "X221", 47.147, 7.264], ["Biel", "X222", 47.147, 7.244], ["Biel", "X223", 47.126, 7.257], ["Biel", "X224", 47.151, 7.271], ["Biel", "X225", 47.159, 7.278], ["Biel", "X226", 47.122, 7.251], ["Biel", "X227", 47.15, 7.261], ["Biel", "X228", 47.142, 7.271], ["Biel", "X229", 47.136, 7.239], ["Biel", "X230", 47.131, 7.221], ["Biel", "X231", 47.126, 7.232], ["Biel", "X232", 47.144, 7.256], ["Biel", "X233", 47.133, 7.249], ["Biel", "X234", 47.159, 7.292], ["Biel", "X235", 47.151, 7.257], ["Biel", "X236", 47.154, 7.268], ["Biel", "X237", 47.163, 7.292], ["Biel", "X238", 47.139, 7.242], ["Biel", "X239", 47.125, 7.233], ["Biel", "X240", 47.128, 7.266], ["NoviSad", "A.1", 45.295, 19.886], ["NoviSad", "D.1", 45.385, 19.831], ["NoviSad", "s10.1", 45.267, 19.884], ["NoviSad", "s2.1", 45.258, 19.815], ["NoviSad", "s2.2", 45.249, 19.837], ["NoviSad", "s2.3", 45.261, 19.849], ["NoviSad", "s3.1", 45.243, 19.806], ["NoviSad", "s3.2", 45.233, 19.81], ["NoviSad", "s5.1", 45.252, 19.798], ["NoviSad", "s5.2", 45.25, 19.816], ["NoviSad", "s5.3", 45.263, 19.826], ["NoviSad", "s5.4", 45.238, 19.833], ["NoviSad", "s5.5", 45.253, 19.848], ["NoviSad", "s5.6", 45.242, 19.847], ["NoviSad", "s6.1", 45.255, 19.763], ["NoviSad", "s6.2", 45.277, 19.781], ["NoviSad", "s6.3", 45.243, 19.782], ["NoviSad", "s6.4", 45.234, 19.792], ["NoviSad", "s6.5", 45.287, 19.828], ["NoviSad", "s6.6", 45.299, 19.823], ["NoviSad", "s6.7", 45.222, 19.846], ["NoviSad", "s6.8", 45.251, 19.876], ["NoviSad", "s6.9", 45.241, 19.881], ["NoviSad", "s8.1", 45.272, 19.821], ["NoviSad", "s9.1", 45.235, 19.768], ["NoviSad", "s9.2", 45.299, 19.812], ["Ghent", "Honda", 51.09, 3.749], ["Ghent", "Melle", 50.98, 3.816], ["Ghent", "Plantentuin", 51.036, 3.722], ["Ghent", "Provinciehuis", 51.051, 3.728], ["Ghent", "Sintbavo", 51.052, 3.732], ["Ghent", "Wondelgem", 51.084, 3.703], ["Rennes", "SM14", 48.205, -1.681], ["Rennes", "SM15", 48.118, -1.703], ["Rennes", "SM16", 48.115, -1.666], ["Rennes", "SM18", 48.157, -1.634], ["Rennes", "SM19", 48.111, -1.684], ["Rennes", "SM21", 48.088, -1.642], ["Rennes", "SM22", 48.091, -1.662], ["Rennes", "SM23", 48.182, -1.622], ["Rennes", "SM24", 48.144, -1.632], ["Rennes", "SM25", 48.126, -1.642], ["Rennes", "SM26", 48.104, -1.649], ["Rennes", "SM27", 48.103, -1.705], ["Rennes", "SM28", 48.085, -1.686], ["Rennes", "SM29", 48.098, -1.683], ["Rennes", "SM30", 48.079, -1.665], ["Rennes", "SM32", 48.069, -1.736], ["Rennes", "SM33", 48.105, -1.674], ["Rennes", "SM35", 48.117, -1.688], ["Rennes", "SM36", 48.12, -1.753], ["Rennes", "SM37", 48.118, -1.64], ["Rennes", "SM42", 48.108, -1.613], ["Rennes", "SM53", 48.189, -1.545], ["Rennes", "SM54", 48.195, -1.58]];
const EP = ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter'];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const R = 500, STEP = 15;
const CLS = {w:3, b:2, v:1};
function cls(t){
  if (t.building) return 'b';
  if (t.natural==='water' || t.waterway==='riverbank' || ['basin','reservoir'].includes(t.landuse) || t.water) return 'w';
  if (['wood','scrub','grassland','heath','wetland'].includes(t.natural) || ['forest','grass','meadow','recreation_ground','village_green','orchard','vineyard','allotments','cemetery','farmland'].includes(t.landuse) || ['park','garden','pitch','golf_course','nature_reserve','playground'].includes(t.leisure)) return 'v';
  return null;
}
function inPoly(x,y,P){ let c=false; for(let i=0,j=P.length-1;i<P.length;j=i++){ const xi=P[i][0],yi=P[i][1],xj=P[j][0],yj=P[j][1]; if(((yi>y)!==(yj>y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi)) c=!c; } return c; }
async function ovp(q){
  for (let a=0;a<6;a++){
    const ep = EP[a%2];
    try {
      const r = await fetch(ep,{method:'POST',body:'data='+encodeURIComponent(q),headers:{'Content-Type':'application/x-www-form-urlencoded'}});
      if (r.ok) return await r.json();
      window.__log.push(ep+' '+r.status);
    } catch(e){ window.__log.push(ep+' '+e); }
    await sleep(3000*(a+1));
  }
  return null;
}
function qry(lat,lon){
  const a = `(around:${R+30},${lat},${lon})`;
  return `[out:json][timeout:120];(way["building"]${a};way["natural"~"^(water|wood|scrub|grassland|heath|wetland)$"]${a};way["landuse"~"^(forest|grass|meadow|recreation_ground|village_green|orchard|vineyard|allotments|cemetery|farmland|basin|reservoir)$"]${a};way["leisure"~"^(park|garden|pitch|golf_course|nature_reserve|playground)$"]${a};way["waterway"="riverbank"]${a};way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|pedestrian)$"]${a};relation["natural"="water"]${a};relation["landuse"~"^(forest|grass|meadow|basin|reservoir)$"]${a};relation["leisure"~"^(park|garden)$"]${a};relation["waterway"="riverbank"]${a};);out geom;`;
}
async function run(){
  // elevations
  const elev = {};
  for (let i=0;i<S.length;i+=90){
    const ch = S.slice(i,i+90);
    const u = 'https://api.open-meteo.com/v1/elevation?latitude='+ch.map(s=>s[2]).join(',')+'&longitude='+ch.map(s=>s[3]).join(',');
    try { const j = await (await fetch(u)).json(); ch.forEach((s,k)=>elev[s[1]+'|'+s[0]]=j.elevation[k]); } catch(e){ window.__log.push('elev '+e); }
  }
  for (const s of S){
    const [city,id,lat,lon] = s;
    const key = id+'|'+city;
    const j = await ovp(qry(lat,lon));
    if (!j){ window.__res[key] = {city,id,lat,lon,elev:elev[key],err:'overpass_failed'}; window.__n++; continue; }
    const kx = 111320*Math.cos(lat*Math.PI/180), ky = 110540;
    const polys = [], roads = [];
    for (const e of j.elements){
      const t = e.tags||{};
      const ringsOf = [];
      if (e.type==='way' && e.geometry){
        if (t.highway){ roads.push(e.geometry.map(p=>[(p.lon-lon)*kx,(p.lat-lat)*ky])); continue; }
        ringsOf.push(e.geometry);
      } else if (e.type==='relation' && e.members){
        for (const m of e.members){ if (m.role==='outer' && m.geometry && m.geometry.length>3){ const g=m.geometry; if (g[0].lat===g[g.length-1].lat && g[0].lon===g[g.length-1].lon) ringsOf.push(g); } }
      }
      const c = cls(t); if (!c) continue;
      for (const g of ringsOf){
        if (g.length<4) continue;
        const P = g.map(p=>[(p.lon-lon)*kx,(p.lat-lat)*ky]);
        let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9; for (const p of P){ if(p[0]<x0)x0=p[0]; if(p[0]>x1)x1=p[0]; if(p[1]<y0)y0=p[1]; if(p[1]>y1)y1=p[1]; }
        polys.push({c, P, x0,x1,y0,y1});
      }
    }
    // grid
    const bands = [100,250,500];
    const cnt = {}; for (const b of bands){ cnt[b]={n:0,b:0,v:0,w:0}; }
    const sec = Array.from({length:8},()=>({n:0,b:0,v:0,w:0}));
    for (let x=-R+STEP/2; x<R; x+=STEP) for (let y=-R+STEP/2; y<R; y+=STEP){
      const d = Math.hypot(x,y); if (d>R) continue;
      let best = null, bp = 0;
      for (const p of polys){
        if (x<p.x0||x>p.x1||y<p.y0||y>p.y1) continue;
        if (CLS[p.c] > bp && inPoly(x,y,p.P)){ bp = CLS[p.c]; best = p.c; if (bp===3) break; }
      }
      for (const b of bands) if (d<=b){ cnt[b].n++; if (best) cnt[b][best]++; }
      if (d>=50){ let ang = Math.atan2(x,y)*180/Math.PI; if (ang<0) ang+=360; const k = Math.floor(((ang+22.5)%360)/45); sec[k].n++; if (best) sec[k][best]++; }
    }
    const rd = {}; for (const b of bands) rd[b]=0;
    for (const L of roads){ for (let i=1;i<L.length;i++){ const mx=(L[i][0]+L[i-1][0])/2, my=(L[i][1]+L[i-1][1])/2; const d=Math.hypot(mx,my); const len=Math.hypot(L[i][0]-L[i-1][0],L[i][1]-L[i-1][1]); for (const b of bands) if (d<=b) rd[b]+=len; } }
    const f = {};
    for (const b of bands){ const n=cnt[b].n||1; f['bld_'+b]=cnt[b].b/n; f['veg_'+b]=cnt[b].v/n; f['wat_'+b]=cnt[b].w/n; f['road_'+b]=rd[b]/(Math.PI*b*b)*1000; }
    sec.forEach((s,k)=>{ const n=s.n||1; f['sbld_'+k]=s.b/n; f['sveg_'+k]=s.v/n; f['swat_'+k]=s.w/n; });
    window.__res[key] = {city,id,lat,lon,elev:elev[key],f,nPoly:polys.length,nRoad:roads.length};
    window.__n++;
    await sleep(1500);
  }
  window.__done = true;
}
run();
return 'started '+S.length;
})()
