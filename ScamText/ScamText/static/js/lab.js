(function(){
  function el(id){return document.getElementById(id)}
  function post(url,body){
    return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      .then(function(r){return r.json().then(function(d){return {ok:r.ok,d:d}})});
  }
  function pct(x){return (x*100).toFixed(1)+'%'}
  function curve(svg,vals){
    if(!svg||!vals||vals.length<2) return;
    var min=Math.min.apply(null,vals), max=Math.max.apply(null,vals), span=(max-min)||1, pts=[];
    for(var i=0;i<vals.length;i++){pts.push((i/(vals.length-1)*296+2).toFixed(1)+','+(96-(vals[i]-min)/span*92).toFixed(1))}
    svg.textContent='';
    var pl=document.createElementNS('http://www.w3.org/2000/svg','polyline');
    pl.setAttribute('points',pts.join(' ')); pl.setAttribute('fill','none'); pl.setAttribute('stroke','#8b5cf6'); pl.setAttribute('stroke-width','2');
    svg.appendChild(pl);
  }
  function verdict(node,label){node.textContent=label.toUpperCase();node.className='verdict '+(label==='Spam'?'spam':'ham')}
  function chips(node,words,cls){
    node.textContent='';
    words.forEach(function(w){var s=document.createElement('span');s.className='chip'+(cls?' '+cls(w):'');s.textContent=w.text;node.appendChild(s)});
  }

  // ---- curve from the training report
  var cd=el('nn-curve-data'); if(cd){try{curve(el('nn-curve'),JSON.parse(cd.textContent))}catch(e){}}

  // ---- 1) three-model comparison
  var ta=el('lab-msg');
  if(ta){
    var max=parseInt(ta.dataset.max,10), err=el('lab-err'), res=el('lab-res'), prior=el('prior');
    function showErr(box,out,t){box.textContent=t;box.style.display='block';if(out)out.style.display='none'}
    function upd(){el('lab-cnt').textContent=ta.value.length+' / '+max}
    function updPrior(){el('prior-val').textContent=prior.value+'%'}
    ta.addEventListener('input',upd); prior.addEventListener('input',updPrior); upd(); updPrior();
    Array.prototype.forEach.call(document.querySelectorAll('.examples button'),function(b){
      b.addEventListener('click',function(){ta.value=b.getAttribute('data-text');upd()})});
    el('lab-run').addEventListener('click',function(){
      err.style.display='none';
      if(!ta.value.trim()){showErr(err,res,'Мәтін бос. Тексеру үшін хабарлама енгізіңіз.');return}
      if(ta.value.length>max){showErr(err,res,'Мәтін тым ұзын. Ең көбі '+max+' таңба.');return}
      var btn=el('lab-run'); btn.disabled=true;
      post('/api/compare',{text:ta.value,prior:parseInt(prior.value,10)/100}).then(function(o){
        if(!o.ok){showErr(err,res,o.d.error||'Қате орын алды.');return}
        var d=o.d, lr=d.logistic_regression, nb=d.naive_bayes, nn=d.neural_net;
        verdict(el('lr-v'),lr.label); el('lr-p').textContent=pct(lr.spam_probability); el('lr-bar').style.width=Math.round(lr.spam_probability*100)+'%';
        chips(el('lr-w'),lr.suspicious_words.map(function(w){return {text:w}}));
        verdict(el('nn-v'),nn.label); el('nn-p').textContent=pct(nn.spam_probability); el('nn-bar').style.width=Math.round(nn.spam_probability*100)+'%';
        chips(el('nn-w'),nn.suspicious_words.map(function(w){return {text:w}}));
        verdict(el('nb-v'),nb.label);
        var post_=nb.posterior>=0.9999?'>99,99%':nb.posterior<=0.0001?'<0,01%':pct(nb.posterior);
        el('nb-f').textContent='prior odds = '+nb.prior_odds+'  (P(Spam)='+Math.round(nb.prior*100)+'%)\n× 10^'+nb.log10_bayes_factor+' (сөздердің дәлелі)\n⇒ P(Spam | мәтін) = '+post_;
        el('nb-h').textContent='Танылған токен: '+nb.tokens_known+' / '+nb.tokens_total+(nb.tokens_unknown?' (танылмағаны ескерілмейді)':'');
        chips(el('nb-w'),nb.evidence.map(function(e){return {text:e.token+' '+(e.llr>0?'+':'')+e.llr,llr:e.llr}}),function(w){return w.llr>0?'up':'down'});
        res.style.display='block';
      }).catch(function(){showErr(err,res,'Серверге қосылу мүмкін болмады. Қолданба жұмыс істеп тұрғанын тексеріңіз.')})
      .finally(function(){btn.disabled=false});
    });
    // ---- 4) safety attack on the same text
    var sErr=el('s-err'), sRes=el('s-res');
    el('s-run').addEventListener('click',function(){
      sErr.style.display='none';
      if(!ta.value.trim()){showErr(sErr,sRes,'Алдымен жоғарыдағы өріске хабарлама жазыңыз.');return}
      if(ta.value.length>max){showErr(sErr,sRes,'Мәтін тым ұзын. Ең көбі '+max+' таңба.');return}
      var btn=el('s-run'); btn.disabled=true;
      post('/api/safety',{text:ta.value,attack:el('s-attack').value}).then(function(o){
        if(!o.ok){showErr(sErr,sRes,o.d.error||'Қате орын алды.');return}
        var d=o.d; el('s-text').textContent=d.changed?d.attacked_text:'(мәтін өзгермеді: бұл шабуыл осы мәтінге қолданылмайды)';
        var body=el('s-body'); body.textContent='';
        d.results.forEach(function(r){
          var tr=document.createElement('tr'), flipped=r.label_before!==r.label_after;
          [r.model,pct(r.before),pct(r.after),r.label_before+' → '+r.label_after+(flipped?' (алданды)':'')].forEach(function(t,i){
            var td=document.createElement('td'); td.textContent=t; if(i===3)td.className=flipped?'bad':'good'; tr.appendChild(td)});
          body.appendChild(tr)});
        sRes.style.display='block';
      }).catch(function(){showErr(sErr,sRes,'Серверге қосылу мүмкін болмады.')})
      .finally(function(){btn.disabled=false});
    });
  }

  // ---- 3) XOR
  var xr=el('x-run');
  if(xr){
    xr.addEventListener('click',function(){
      var err=el('x-err'), out=el('x-res'); err.style.display='none';
      var body={hidden:Number(el('x-h').value),lr:Number(el('x-lr').value),epochs:Number(el('x-ep').value),seed:Number(el('x-seed').value)};
      xr.disabled=true;
      post('/api/xor',body).then(function(o){
        if(!o.ok){err.textContent=o.d.error||'Қате орын алды.';err.style.display='block';out.style.display='none';return}
        var d=o.d;
        function rows(pred,prob){return d.inputs.map(function(x,i){return '('+x.join(',')+') → '+pred[i]+(prob?' (p='+prob[i]+')':'')+'  [дұрыс: '+d.targets[i]+']'}).join('\n')}
        el('x-pa').textContent=Math.round(d.perceptron.accuracy*100)+'%'; el('x-pp').style.whiteSpace='pre'; el('x-pp').textContent=rows(d.perceptron.predictions);
        el('x-na').textContent=Math.round(d.neural_net.accuracy*100)+'%'; el('x-np').style.whiteSpace='pre'; el('x-np').textContent=rows(d.neural_net.predictions,d.neural_net.probabilities);
        curve(el('x-curve'),d.neural_net.loss_curve); out.style.display='block';
      }).catch(function(){err.textContent='Серверге қосылу мүмкін болмады.';err.style.display='block'})
      .finally(function(){xr.disabled=false});
    });
  }
})();
