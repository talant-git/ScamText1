(function(){
  var ta=document.getElementById('msg'); if(!ta) return;
  var cnt=document.getElementById('cnt'), btn=document.getElementById('check'),
      err=document.getElementById('err'), res=document.getElementById('result'),
      lang=document.getElementById('lang'), max=parseInt(ta.dataset.max,10);
  function upd(){cnt.textContent=ta.value.length+' / '+max}
  ta.addEventListener('input',upd); upd();
  function showErr(t){err.textContent=t;err.style.display='block';res.style.display='none'}
  function el(id){return document.getElementById(id)}
  btn.addEventListener('click',function(){
    err.style.display='none';
    if(!ta.value.trim()){showErr('Мәтін бос. Тексеру үшін хабарлама енгізіңіз.');return}
    if(ta.value.length>max){showErr('Мәтін тым ұзын. Ең көбі '+max+' таңба.');return}
    btn.disabled=true; btn.textContent='Талдануда...';
    fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:ta.value,language:lang.value})})
    .then(function(r){return r.json().then(function(d){return {ok:r.ok,d:d}})})
    .then(function(o){
      if(!o.ok){showErr(o.d.error||'Қате орын алды.');return}
      var d=o.d, v=el('verdict'), spam=d.label==='Spam';
      v.textContent=spam?'SPAM':'HAM'; v.className='verdict '+(spam?'spam':'ham');
      el('rlang').textContent=d.language;
      var pc=Math.round(d.confidence*100); el('rconf').textContent=pc+'%'; el('rbar').style.width=pc+'%';
      var w=el('rwords'); w.textContent='';
      if(d.suspicious_words.length){d.suspicious_words.forEach(function(x){var s=document.createElement('span');s.className='chip';s.textContent=x;w.appendChild(s)});
        el('rhint').textContent=spam?'Модель Spam деп шешуіне ең көп әсер еткен сөздер:':'Модель Ham деп шешуіне ең көп әсер еткен сөздер:'}
      else{el('rhint').textContent='Жеке сөздердің айқын әсері табылмады.'}
      res.style.display='block';
    })
    .catch(function(){showErr('Серверге қосылу мүмкін болмады. Қолданба жұмыс істеп тұрғанын тексеріңіз.')})
    .finally(function(){btn.disabled=false;btn.textContent='Тексеру'});
  });
  el('again').addEventListener('click',function(){ta.value='';upd();res.style.display='none';err.style.display='none';ta.focus()});
})();
