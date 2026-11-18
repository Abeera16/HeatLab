// Fetches raw OpenStreetMap geometries (ways + relations) around every station with the Overpass API, in a browser tab.
// Same tag filters as overpass_features.js, but returns geometry so GeoPandas can do the overlay (scripts/06_vector_features.py).
// Per station: bounding box of +/-0.0051 deg (about 570 m), tags reduced to the ones used for classification.
// Output per city: osm_geom_<City>.json = { "w123": {t: tags, g: [[lon,lat],...]}, "r45": {t: tags, m: [{r: role, g: [...]}]} }
// Usage: paste into the console of any page that allows cross-origin fetch, set S to [[city, station, lat, lon], ...].
// (Implementation used in this project: 4 parallel workers across overpass-api.de, overpass.kumi.systems, overpass.openstreetmap.fr, overpass.osm.ch;
//  15-76 s per station when the public servers are busy; each city is saved to disk as soon as it completes.)

(()=>{
window.__osm={}; window.__log=[]; window.__done=false; window.__n=0; window.__fail=[]; window.__cnt={}; window.__dl=[];
const S=[["Biel", "X201", 47.131, 7.241], ["Biel", "X202", 47.131, 7.236], ["Biel", "X203", 47.137, 7.247], ["Biel", "X206", 47.139, 7.295], ["Biel", "X208", 47.13, 7.262], ["Biel", "X204", 47.141, 7.253], ["Biel", "X209", 47.135, 7.256], ["Biel", "X207", 47.179, 7.415], ["Biel", "X211", 47.128, 7.249], ["Biel", "X210", 47.135, 7.249], ["Biel", "X205", 47.145, 7.265], ["Biel", "X214", 47.128, 7.266], ["Biel", "X213", 47.12, 7.254], ["Biel", "X217", 47.144, 7.284], ["Biel", "X215", 47.142, 7.246], ["Biel", "X218", 47.133, 7.243], ["Biel", "X212", 47.134, 7.231], ["Biel", "X220", 47.137, 7.251], ["Biel", "X222", 47.147, 7.244], ["Biel", "X221", 47.147, 7.264], ["Biel", "X224", 47.151, 7.271], ["Biel", "X223", 47.126, 7.257], ["Biel", "X225", 47.159, 7.278], ["Biel", "X226", 47.122, 7.251], ["Biel", "X227", 47.15, 7.261], ["Biel", "X229", 47.136, 7.239], ["Biel", "X228", 47.142, 7.271], ["Biel", "X230", 47.131, 7.221], ["Biel", "X219", 47.139, 7.246], ["Biel", "X231", 47.126, 7.232], ["Biel", "X232", 47.144, 7.256], ["Biel", "X235", 47.151, 7.257], ["Biel", "X234", 47.159, 7.292], ["Biel", "X233", 47.133, 7.249], ["Biel", "X236", 47.154, 7.268], ["Biel", "X216", 47.125, 7.239], ["Biel", "X237", 47.163, 7.292], ["Biel", "X238", 47.139, 7.242], ["Biel", "X239", 47.125, 7.233], ["Biel", "X240", 47.128, 7.266], ["NoviSad", "A.1", 45.295, 19.886], ["NoviSad", "D.1", 45.385, 19.831], ["NoviSad", "s10.1", 45.267, 19.884], ["NoviSad", "s2.1", 45.258, 19.815], ["NoviSad", "s2.2", 45.249, 19.837], ["NoviSad", "s2.3", 45.261, 19.849], ["NoviSad", "s3.1", 45.243, 19.806], ["NoviSad", "s3.2", 45.233, 19.81], ["NoviSad", "s5.1", 45.252, 19.798], ["NoviSad", "s5.2", 45.25, 19.816], ["NoviSad", "s5.3", 45.263, 19.826], ["NoviSad", "s5.4", 45.238, 19.833], ["NoviSad", "s5.5", 45.253, 19.848], ["NoviSad", "s5.6", 45.242, 19.847], ["NoviSad", "s6.1", 45.255, 19.763], ["NoviSad", "s6.2", 45.277, 19.781], ["NoviSad", "s6.3", 45.243, 19.782], ["NoviSad", "s6.4", 45.234, 19.792], ["NoviSad", "s6.5", 45.287, 19.828], ["NoviSad", "s6.6", 45.299, 19.823], ["NoviSad", "s6.7", 45.222, 19.846], ["NoviSad", "s6.8", 45.251, 19.876], ["NoviSad", "s6.9", 45.241, 19.881], ["NoviSad", "s8.1", 45.272, 19.821], ["NoviSad", "s9.1", 45.235, 19.768], ["NoviSad", "s9.2", 45.299, 19.812], ["Ghent", "Honda", 51.09, 3.749], ["Ghent", "Melle", 50.98, 3.816], ["Ghent", "Plantentuin", 51.036, 3.722], ["Rennes", "SM14", 48.205, -1.681], ["Ghent", "Provinciehuis", 51.051, 3.728], ["Ghent", "Wondelgem", 51.084, 3.703], ["Ghent", "Sintbavo", 51.052, 3.732], ["Rennes", "SM15", 48.118, -1.703], ["Rennes", "SM18", 48.157, -1.634], ["Rennes", "SM16", 48.115, -1.666], ["Rennes", "SM19", 48.111, -1.684], ["Rennes", "SM21", 48.088, -1.642], ["Rennes", "SM23", 48.182, -1.622], ["Rennes", "SM22", 48.091, -1.662], ["Rennes", "SM24", 48.144, -1.632], ["Rennes", "SM25", 48.126, -1.642], ["Rennes", "SM26", 48.104, -1.649], ["Rennes", "SM28", 48.085, -1.686], ["Rennes", "SM27", 48.103, -1.705], ["Rennes", "SM32", 48.069, -1.736], ["Rennes", "SM29", 48.098, -1.683], ["Rennes", "SM30", 48.079, -1.665], ["Rennes", "SM33", 48.105, -1.674], ["Rennes", "SM35", 48.117, -1.688], ["Rennes", "SM36", 48.12, -1.753], ["Rennes", "SM37", 48.118, -1.64], ["Rennes", "SM53", 48.189, -1.545], ["Rennes", "SM42", 48.108, -1.613], ["Rennes", "SM54", 48.195, -1.58]];
const TOT={}; S.forEach(s=>TOT[s[0]]=(TOT[s[0]]||0)+1);
const DE='https://overpass-api.de/api/interpreter',KU='https://overpass.kumi.systems/api/interpreter',FR='https://overpass.openstreetmap.fr/api/interpreter',CH='https://overpass.osm.ch/api/interpreter';
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function qry(lat,lon){const dl=0.0051,dn=0.0051/Math.cos(lat*Math.PI/180);const a=`(${(lat-dl).toFixed(5)},${(lon-dn).toFixed(5)},${(lat+dl).toFixed(5)},${(lon+dn).toFixed(5)})`;return `[out:json][timeout:60];(way["building"]${a};way["natural"~"^(water|wood|scrub|grassland|heath|wetland)$"]${a};way["landuse"~"^(forest|grass|meadow|recreation_ground|village_green|orchard|vineyard|allotments|cemetery|farmland|basin|reservoir)$"]${a};way["leisure"~"^(park|garden|pitch|golf_course|nature_reserve|playground)$"]${a};way["waterway"="riverbank"]${a};way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|pedestrian)$"]${a};relation["natural"="water"]${a};relation["landuse"~"^(forest|grass|meadow|basin|reservoir)$"]${a};relation["leisure"~"^(park|garden)$"]${a};relation["waterway"="riverbank"]${a};);out geom;`;}
async function ovp(q,eps){for(let a=0;a<16;a++){const ep=eps[a%eps.length];const ac=new AbortController();const to=setTimeout(()=>ac.abort(),75000);try{const r=await fetch(ep,{method:'POST',body:'data='+encodeURIComponent(q),headers:{'Content-Type':'application/x-www-form-urlencoded'},signal:ac.signal});if(r.ok){const j=await r.json();clearTimeout(to);if(j.elements)return j;}window.__log.push(ep.slice(8,20)+' '+r.status);}catch(e){window.__log.push(ep.slice(8,20)+' '+(e.name||e));}clearTimeout(to);await sleep(Math.min(15000,2000*(a+1)));}return null;}
const KEYS=['building','natural','landuse','leisure','waterway','highway','water'];
const slim=t=>{const o={};for(const k of KEYS)if(t[k]!==undefined)o[k]=t[k];return o;};
const rd=g=>g.map(p=>[+p.lon.toFixed(6),+p.lat.toFixed(6)]);
function save(city){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(window.__osm[city])],{type:'application/json'}));a.download='osm_geom_'+city+'.json';document.body.appendChild(a);a.click();window.__dl.push(city);}
let p=0;
async function worker(eps){
 while(p<S.length){
  const [city,id,lat,lon]=S[p++];
  const j=await ovp(qry(lat,lon),eps);
  if(!j){window.__fail.push(city+'|'+id);}
  else{const c=window.__osm[city]||(window.__osm[city]={});
   for(const e of j.elements){const k=e.type[0]+e.id;if(c[k])continue;
    if(e.type==='way'&&e.geometry)c[k]={t:slim(e.tags||{}),g:rd(e.geometry)};
    else if(e.type==='relation'&&e.members)c[k]={t:slim(e.tags||{}),m:e.members.filter(m=>m.geometry).map(m=>({r:m.role,g:rd(m.geometry)}))};}}
  window.__n++; window.__cnt[city]=(window.__cnt[city]||0)+1;
  if(window.__cnt[city]===TOT[city]) save(city);
  await sleep(300);
 }
}
Promise.all([worker([FR,DE]),worker([DE,FR]),worker([KU,FR]),worker([FR,KU,CH])]).then(()=>{window.__done=true;});
return 'started '+S.length;
})()
