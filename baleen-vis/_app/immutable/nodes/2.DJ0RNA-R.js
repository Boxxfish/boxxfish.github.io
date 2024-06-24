import{s as it,n as Be,f as qe}from"../chunks/scheduler.Cv2zyRbU.js";import{S as ot,i as at,e as v,s as q,m as He,y as ht,c as d,g as a,h as z,z as G,d as A,o as b,k as g,j as p,A as Z,b as B,f as H,B as Me,C as st,l as J}from"../chunks/index.CBmL8D5r.js";import{g as ct}from"../chunks/entry.C4DHhAO8.js";function P(n){return(n==null?void 0:n.length)!==void 0?n:Array.from(n)}const ft=async({fetch:n,params:l})=>{const i=await(await n("./queries.json")).json(),o=l.slug.split(".")[0];return{query_data:await(await n(`./data/${o}.json`)).json(),all_queries:new Map(Object.entries(i)),query_idx:parseInt(o)}},rt=()=>[...Array(10).keys()].map(l=>({slug:`${l}`})),gt=Object.freeze(Object.defineProperty({__proto__:null,entries:rt,load:ft},Symbol.toStringTag,{value:"Module"}));function Fe(n,l,s){const i=n.slice();return i[7]=l[s],i}function Le(n,l,s){const i=n.slice();return i[10]=l[s],i}function Oe(n,l,s){const i=n.slice();return i[10]=l[s],i}function ye(n,l,s){const i=n.slice();return i[15]=l[s],i[17]=s,i}function Ke(n,l,s){const i=n.slice();return i[18]=l[s],i}function Qe(n,l,s){const i=n.slice();return i[18]=l[s],i}function Re(n,l,s){const i=n.slice();return i[23]=l[s],i[25]=s,i}function Ue(n,l,s){const i=n.slice();return i[25]=l[s],i}function Ge(n,l,s){const i=n.slice();return i[28]=l[s][0],i[29]=l[s][1],i}function Je(n){let l,s=n[29]+"",i,o,c,h,r,u;function f(){return n[5](n[28])}return{c(){l=v("option"),i=B(s),this.h()},l(k){l=d(k,"OPTION",{id:!0});var m=A(l);i=H(m,s),m.forEach(a),this.h()},h(){b(l,"id",o=n[28]),l.selected=c=n[28]===n[0].query_idx.toString(),l.__value=h=n[29],Me(l,l.__value)},m(k,m){p(k,l,m),g(l,i),r||(u=st(l,"click",f),r=!0)},p(k,m){n=k,m[0]&8&&s!==(s=n[29]+"")&&J(i,s),m[0]&8&&o!==(o=n[28])&&b(l,"id",o),m[0]&9&&c!==(c=n[28]===n[0].query_idx.toString())&&(l.selected=c),m[0]&8&&h!==(h=n[29])&&(l.__value=h,Me(l,l.__value))},d(k){k&&a(l),r=!1,u()}}}function We(n){let l,s=n[25]+1+"",i,o,c,h;function r(){return n[6](n[25])}return{c(){l=v("button"),i=B(s),this.h()},l(u){l=d(u,"BUTTON",{class:!0});var f=A(l);i=H(f,s),f.forEach(a),this.h()},h(){b(l,"class",o=qe(n[1]==n[25]?"step-selected":"")+" svelte-63wgtz")},m(u,f){p(u,l,f),g(l,i),c||(h=st(l,"click",r),c=!0)},p(u,f){n=u,f[0]&2&&o!==(o=qe(n[1]==n[25]?"step-selected":"")+" svelte-63wgtz")&&b(l,"class",o)},d(u){u&&a(l),c=!1,h()}}}function Xe(n){let l,s=n[23]+"",i;return{c(){l=v("div"),i=B(s),this.h()},l(o){l=d(o,"DIV",{class:!0});var c=A(l);i=H(c,s),c.forEach(a),this.h()},h(){b(l,"class","token "+(n[25]<64?"token-q":"token-c")+" svelte-63wgtz")},m(o,c){p(o,l,c),g(l,i)},p(o,c){c[0]&4&&s!==(s=o[23]+"")&&J(i,s)},d(o){o&&a(l)}}}function Ye(n){let l,s=n[2].c_toks[n[18]]+"",i;return{c(){l=v("div"),i=B(s),this.h()},l(o){l=d(o,"DIV",{class:!0});var c=A(l);i=H(c,s),c.forEach(a),this.h()},h(){b(l,"class","token token-q svelte-63wgtz")},m(o,c){p(o,l,c),g(l,i)},p(o,c){c[0]&4&&s!==(s=o[2].c_toks[o[18]]+"")&&J(i,s)},d(o){o&&a(l)}}}function Ze(n){let l,s="Context Token Matches (8)",i,o,c=P(n[2].c_matches[n[17]]),h=[];for(let r=0;r<c.length;r+=1)h[r]=$e(Ke(n,c,r));return{c(){l=v("span"),l.textContent=s,i=q(),o=v("div");for(let r=0;r<h.length;r+=1)h[r].c();this.h()},l(r){l=d(r,"SPAN",{"data-svelte-h":!0}),G(l)!=="svelte-19peww2"&&(l.textContent=s),i=z(r),o=d(r,"DIV",{class:!0});var u=A(o);for(let f=0;f<h.length;f+=1)h[f].l(u);u.forEach(a),this.h()},h(){b(o,"class","token-list svelte-63wgtz")},m(r,u){p(r,l,u),p(r,i,u),p(r,o,u);for(let f=0;f<h.length;f+=1)h[f]&&h[f].m(o,null)},p(r,u){if(u[0]&4){c=P(r[2].c_matches[r[17]]);let f;for(f=0;f<c.length;f+=1){const k=Ke(r,c,f);h[f]?h[f].p(k,u):(h[f]=$e(k),h[f].c(),h[f].m(o,null))}for(;f<h.length;f+=1)h[f].d(1);h.length=c.length}},d(r){r&&(a(l),a(i),a(o)),Z(h,r)}}}function $e(n){let l,s=n[2].c_toks[64+n[18]]+"",i;return{c(){l=v("div"),i=B(s),this.h()},l(o){l=d(o,"DIV",{class:!0});var c=A(l);i=H(c,s),c.forEach(a),this.h()},h(){b(l,"class","token token-c svelte-63wgtz")},m(o,c){p(o,l,c),g(l,i)},p(o,c){c[0]&4&&s!==(s=o[2].c_toks[64+o[18]]+"")&&J(i,s)},d(o){o&&a(l)}}}function et(n){let l,s,i=n[15][0].toFixed(2)+"",o,c,h,r=n[15][1]+"",u,f,k,m,N,T="Query Token Matches (32)",C,E,Q,R=P(n[2].q_matches[n[17]]),I=[];for(let w=0;w<R.length;w+=1)I[w]=Ye(Qe(n,R,w));let D=n[2].c_matches[n[17]].length>0&&Ze(n);return{c(){l=v("p"),s=v("b"),o=B(i),c=B(`:
      `),h=v("a"),u=B(r),k=q(),m=v("div"),N=v("span"),N.textContent=T,C=q(),E=v("div");for(let w=0;w<I.length;w+=1)I[w].c();Q=q(),D&&D.c(),this.h()},l(w){l=d(w,"P",{});var j=A(l);s=d(j,"B",{});var S=A(s);o=H(S,i),c=H(S,`:
      `),h=d(S,"A",{href:!0,class:!0});var Y=A(h);u=H(Y,r),Y.forEach(a),S.forEach(a),j.forEach(a),k=z(w),m=d(w,"DIV",{});var U=A(m);N=d(U,"SPAN",{"data-svelte-h":!0}),G(N)!=="svelte-182sf6i"&&(N.textContent=T),C=z(U),E=d(U,"DIV",{class:!0});var ne=A(E);for(let X=0;X<I.length;X+=1)I[X].l(ne);ne.forEach(a),Q=z(U),D&&D.l(U),U.forEach(a),this.h()},h(){b(h,"href",f="https://en.wikipedia.org/wiki/"+n[15][1].split("|")[0].replace(" ","_")),b(h,"class","svelte-63wgtz"),b(E,"class","token-list svelte-63wgtz")},m(w,j){p(w,l,j),g(l,s),g(s,o),g(s,c),g(s,h),g(h,u),p(w,k,j),p(w,m,j),g(m,N),g(m,C),g(m,E);for(let S=0;S<I.length;S+=1)I[S]&&I[S].m(E,null);g(m,Q),D&&D.m(m,null)},p(w,j){if(j[0]&4&&i!==(i=w[15][0].toFixed(2)+"")&&J(o,i),j[0]&4&&r!==(r=w[15][1]+"")&&J(u,r),j[0]&4&&f!==(f="https://en.wikipedia.org/wiki/"+w[15][1].split("|")[0].replace(" ","_"))&&b(h,"href",f),j[0]&4){R=P(w[2].q_matches[w[17]]);let S;for(S=0;S<R.length;S+=1){const Y=Qe(w,R,S);I[S]?I[S].p(Y,j):(I[S]=Ye(Y),I[S].c(),I[S].m(E,null))}for(;S<I.length;S+=1)I[S].d(1);I.length=R.length}w[2].c_matches[w[17]].length>0?D?D.p(w,j):(D=Ze(w),D.c(),D.m(m,null)):D&&(D.d(1),D=null)},d(w){w&&(a(l),a(k),a(m)),Z(I,w),D&&D.d()}}}function tt(n){let l,s,i=n[10][0].toFixed(2)+"",o,c,h,r=n[10][1].split("|")[0]+"",u,f,k,m=n[10][1].split("|")[1]+"",N;return{c(){l=v("p"),s=v("b"),o=B(i),c=B(`:
      `),h=v("a"),u=B(r),k=q(),N=B(m),this.h()},l(T){l=d(T,"P",{});var C=A(l);s=d(C,"B",{});var E=A(s);o=H(E,i),c=H(E,`:
      `),h=d(E,"A",{href:!0,class:!0});var Q=A(h);u=H(Q,r),Q.forEach(a),E.forEach(a),k=z(C),N=H(C,m),C.forEach(a),this.h()},h(){b(h,"href",f="https://en.wikipedia.org/wiki/"+n[10][1].split("|")[0].replace(" ","_")),b(h,"class","svelte-63wgtz")},m(T,C){p(T,l,C),g(l,s),g(s,o),g(s,c),g(s,h),g(h,u),g(l,k),g(l,N)},p(T,C){C[0]&4&&i!==(i=T[10][0].toFixed(2)+"")&&J(o,i),C[0]&4&&r!==(r=T[10][1].split("|")[0]+"")&&J(u,r),C[0]&4&&f!==(f="https://en.wikipedia.org/wiki/"+T[10][1].split("|")[0].replace(" ","_"))&&b(h,"href",f),C[0]&4&&m!==(m=T[10][1].split("|")[1]+"")&&J(N,m)},d(T){T&&a(l)}}}function lt(n){let l,s,i=n[10][0].toFixed(2)+"",o,c,h,r=n[10][1].split("|")[0]+"",u,f,k,m=n[10][1].split("|")[1]+"",N,T;return{c(){l=v("p"),s=v("b"),o=B(i),c=B(`:
      `),h=v("a"),u=B(r),k=q(),N=B(m),this.h()},l(C){l=d(C,"P",{class:!0});var E=A(l);s=d(E,"B",{});var Q=A(s);o=H(Q,i),c=H(Q,`:
      `),h=d(Q,"A",{href:!0,class:!0});var R=A(h);u=H(R,r),R.forEach(a),Q.forEach(a),k=z(E),N=H(E,m),E.forEach(a),this.h()},h(){b(h,"href",f="https://en.wikipedia.org/wiki/"+n[10][1].split("|")[0].replace(" ","_")),b(h,"class","svelte-63wgtz"),b(l,"class",T=qe(n[10][0]<0?"condenser-excluded":"")+" svelte-63wgtz")},m(C,E){p(C,l,E),g(l,s),g(s,o),g(s,c),g(s,h),g(h,u),g(l,k),g(l,N)},p(C,E){E[0]&4&&i!==(i=C[10][0].toFixed(2)+"")&&J(o,i),E[0]&4&&r!==(r=C[10][1].split("|")[0]+"")&&J(u,r),E[0]&4&&f!==(f="https://en.wikipedia.org/wiki/"+C[10][1].split("|")[0].replace(" ","_"))&&b(h,"href",f),E[0]&4&&m!==(m=C[10][1].split("|")[1]+"")&&J(N,m),E[0]&4&&T!==(T=qe(C[10][0]<0?"condenser-excluded":"")+" svelte-63wgtz")&&b(l,"class",T)},d(C){C&&a(l)}}}function nt(n){let l,s,i=n[7].split("|")[0]+"",o,c,h=n[7].split("|")[1]+"",r;return{c(){l=v("p"),s=v("b"),o=B(i),c=q(),r=B(h)},l(u){l=d(u,"P",{});var f=A(l);s=d(f,"B",{});var k=A(s);o=H(k,i),k.forEach(a),c=z(f),r=H(f,h),f.forEach(a)},m(u,f){p(u,l,f),g(l,s),g(s,o),g(l,c),g(l,r)},p(u,f){f[0]&4&&i!==(i=u[7].split("|")[0]+"")&&J(o,i),f[0]&4&&h!==(h=u[7].split("|")[1]+"")&&J(r,h)},d(u){u&&a(l)}}}function _t(n){let l,s,i,o,c,h="Baleen Visualizer",r,u,f,k="Query:",m,N,T,C,E="Hop:",Q,R,I,D="Query + Context",w,j,S=`At the start of each hop, the query and context are tokenized and
  contextualized with a BERT-based model. Queries are padded to a length of 64
  with [MASK] tokens.`,Y,U,ne,X,Ie="Retrieval",ge,$,Se=`Each document is scored using the contextualized query. Unlike ColBERT, not
  every token in the query/context is used. Instead, Baleen uses <i>focused</i> late
  interaction, where only the top N most relevant query/context tokens are used for
  MaxSim. The score for the query and the score for the context are computed independently,
  and use a different number of tokens (32 for the query, 8 for the context).`,me,ae,se,Ne="Condenser (Stage 1)",ke,ee,Te=`The first stage reads all passages independently, then assigns a score to each
  sentence within the passage. The top 9 sentences across all passages ove on to
  the next stage.`,Ce,he,ie,Ve="Condenser (Stage 2)",we,te,je=`The second stage reads all sentences from the previous stage at once, then
  rescores all sentences. All positive passages move on.`,be,ce,oe,Ae="New Context",Ee,le,De=`The context for the next hop consists of the facts produced by the previous
  stage. If this is the final hop, the query and context are passed to the
  reader for answering.`,xe,fe,re=P(n[3]),M=[];for(let e=0;e<re.length;e+=1)M[e]=Je(Ge(n,re,e));let ze=P([0,1,2,3]),W=[];for(let e=0;e<4;e+=1)W[e]=We(Ue(n,ze,e));let _e=P(n[2].c_toks),F=[];for(let e=0;e<_e.length;e+=1)F[e]=Xe(Re(n,_e,e));let ue=P(n[2].ranking),L=[];for(let e=0;e<ue.length;e+=1)L[e]=et(ye(n,ue,e));let pe=P(n[2].stage1),O=[];for(let e=0;e<pe.length;e+=1)O[e]=tt(Oe(n,pe,e));let ve=P(n[2].stage2),y=[];for(let e=0;e<ve.length;e+=1)y[e]=lt(Le(n,ve,e));let de=P(n[2].new_ctx.split("[SEP]")),K=[];for(let e=0;e<de.length;e+=1)K[e]=nt(Fe(n,de,e));return{c(){l=v("link"),s=v("link"),i=v("link"),o=q(),c=v("h1"),c.textContent=h,r=q(),u=v("div"),f=v("span"),f.textContent=k,m=q(),N=v("select");for(let e=0;e<M.length;e+=1)M[e].c();T=q(),C=v("span"),C.textContent=E,Q=q();for(let e=0;e<4;e+=1)W[e].c();R=q(),I=v("h2"),I.textContent=D,w=q(),j=v("div"),j.textContent=S,Y=q(),U=v("div");for(let e=0;e<F.length;e+=1)F[e].c();ne=q(),X=v("h2"),X.textContent=Ie,ge=q(),$=v("div"),$.innerHTML=Se,me=q();for(let e=0;e<L.length;e+=1)L[e].c();ae=q(),se=v("h2"),se.textContent=Ne,ke=q(),ee=v("div"),ee.textContent=Te,Ce=q();for(let e=0;e<O.length;e+=1)O[e].c();he=q(),ie=v("h2"),ie.textContent=Ve,we=q(),te=v("div"),te.textContent=je,be=q();for(let e=0;e<y.length;e+=1)y[e].c();ce=q(),oe=v("h2"),oe.textContent=Ae,Ee=q(),le=v("div"),le.textContent=De,xe=q();for(let e=0;e<K.length;e+=1)K[e].c();fe=He(),this.h()},l(e){const _=ht("svelte-pa9zau",document.head);l=d(_,"LINK",{rel:!0,href:!0}),s=d(_,"LINK",{rel:!0,href:!0,crossorigin:!0}),i=d(_,"LINK",{href:!0,rel:!0}),_.forEach(a),o=z(e),c=d(e,"H1",{"data-svelte-h":!0}),G(c)!=="svelte-c58z83"&&(c.textContent=h),r=z(e),u=d(e,"DIV",{});var t=A(u);f=d(t,"SPAN",{"data-svelte-h":!0}),G(f)!=="svelte-rocmfk"&&(f.textContent=k),m=z(t),N=d(t,"SELECT",{});var V=A(N);for(let x=0;x<M.length;x+=1)M[x].l(V);V.forEach(a),t.forEach(a),T=z(e),C=d(e,"SPAN",{"data-svelte-h":!0}),G(C)!=="svelte-1ixxeo9"&&(C.textContent=E),Q=z(e);for(let x=0;x<4;x+=1)W[x].l(e);R=z(e),I=d(e,"H2",{"data-svelte-h":!0}),G(I)!=="svelte-1m5ncis"&&(I.textContent=D),w=z(e),j=d(e,"DIV",{class:!0,"data-svelte-h":!0}),G(j)!=="svelte-11dr055"&&(j.textContent=S),Y=z(e),U=d(e,"DIV",{class:!0});var Pe=A(U);for(let x=0;x<F.length;x+=1)F[x].l(Pe);Pe.forEach(a),ne=z(e),X=d(e,"H2",{"data-svelte-h":!0}),G(X)!=="svelte-1cx83v6"&&(X.textContent=Ie),ge=z(e),$=d(e,"DIV",{class:!0,"data-svelte-h":!0}),G($)!=="svelte-kj5rhx"&&($.innerHTML=Se),me=z(e);for(let x=0;x<L.length;x+=1)L[x].l(e);ae=z(e),se=d(e,"H2",{"data-svelte-h":!0}),G(se)!=="svelte-12qkxrj"&&(se.textContent=Ne),ke=z(e),ee=d(e,"DIV",{class:!0,"data-svelte-h":!0}),G(ee)!=="svelte-1r0tew6"&&(ee.textContent=Te),Ce=z(e);for(let x=0;x<O.length;x+=1)O[x].l(e);he=z(e),ie=d(e,"H2",{"data-svelte-h":!0}),G(ie)!=="svelte-u86rdi"&&(ie.textContent=Ve),we=z(e),te=d(e,"DIV",{class:!0,"data-svelte-h":!0}),G(te)!=="svelte-1mrr67c"&&(te.textContent=je),be=z(e);for(let x=0;x<y.length;x+=1)y[x].l(e);ce=z(e),oe=d(e,"H2",{"data-svelte-h":!0}),G(oe)!=="svelte-1sz1qfl"&&(oe.textContent=Ae),Ee=z(e),le=d(e,"DIV",{class:!0,"data-svelte-h":!0}),G(le)!=="svelte-bzlbk2"&&(le.textContent=De),xe=z(e);for(let x=0;x<K.length;x+=1)K[x].l(e);fe=He(),this.h()},h(){b(l,"rel","preconnect"),b(l,"href","https://fonts.googleapis.com"),b(s,"rel","preconnect"),b(s,"href","https://fonts.gstatic.com"),b(s,"crossorigin","true"),b(i,"href","https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap"),b(i,"rel","stylesheet"),b(j,"class","explanation svelte-63wgtz"),b(U,"class","token-list svelte-63wgtz"),b($,"class","explanation svelte-63wgtz"),b(ee,"class","explanation svelte-63wgtz"),b(te,"class","explanation svelte-63wgtz"),b(le,"class","explanation svelte-63wgtz")},m(e,_){g(document.head,l),g(document.head,s),g(document.head,i),p(e,o,_),p(e,c,_),p(e,r,_),p(e,u,_),g(u,f),g(u,m),g(u,N);for(let t=0;t<M.length;t+=1)M[t]&&M[t].m(N,null);p(e,T,_),p(e,C,_),p(e,Q,_);for(let t=0;t<4;t+=1)W[t]&&W[t].m(e,_);p(e,R,_),p(e,I,_),p(e,w,_),p(e,j,_),p(e,Y,_),p(e,U,_);for(let t=0;t<F.length;t+=1)F[t]&&F[t].m(U,null);p(e,ne,_),p(e,X,_),p(e,ge,_),p(e,$,_),p(e,me,_);for(let t=0;t<L.length;t+=1)L[t]&&L[t].m(e,_);p(e,ae,_),p(e,se,_),p(e,ke,_),p(e,ee,_),p(e,Ce,_);for(let t=0;t<O.length;t+=1)O[t]&&O[t].m(e,_);p(e,he,_),p(e,ie,_),p(e,we,_),p(e,te,_),p(e,be,_);for(let t=0;t<y.length;t+=1)y[t]&&y[t].m(e,_);p(e,ce,_),p(e,oe,_),p(e,Ee,_),p(e,le,_),p(e,xe,_);for(let t=0;t<K.length;t+=1)K[t]&&K[t].m(e,_);p(e,fe,_)},p(e,_){if(_[0]&11){re=P(e[3]);let t;for(t=0;t<re.length;t+=1){const V=Ge(e,re,t);M[t]?M[t].p(V,_):(M[t]=Je(V),M[t].c(),M[t].m(N,null))}for(;t<M.length;t+=1)M[t].d(1);M.length=re.length}if(_[0]&2){ze=P([0,1,2,3]);let t;for(t=0;t<4;t+=1){const V=Ue(e,ze,t);W[t]?W[t].p(V,_):(W[t]=We(V),W[t].c(),W[t].m(R.parentNode,R))}for(;t<4;t+=1)W[t].d(1)}if(_[0]&4){_e=P(e[2].c_toks);let t;for(t=0;t<_e.length;t+=1){const V=Re(e,_e,t);F[t]?F[t].p(V,_):(F[t]=Xe(V),F[t].c(),F[t].m(U,null))}for(;t<F.length;t+=1)F[t].d(1);F.length=_e.length}if(_[0]&4){ue=P(e[2].ranking);let t;for(t=0;t<ue.length;t+=1){const V=ye(e,ue,t);L[t]?L[t].p(V,_):(L[t]=et(V),L[t].c(),L[t].m(ae.parentNode,ae))}for(;t<L.length;t+=1)L[t].d(1);L.length=ue.length}if(_[0]&4){pe=P(e[2].stage1);let t;for(t=0;t<pe.length;t+=1){const V=Oe(e,pe,t);O[t]?O[t].p(V,_):(O[t]=tt(V),O[t].c(),O[t].m(he.parentNode,he))}for(;t<O.length;t+=1)O[t].d(1);O.length=pe.length}if(_[0]&4){ve=P(e[2].stage2);let t;for(t=0;t<ve.length;t+=1){const V=Le(e,ve,t);y[t]?y[t].p(V,_):(y[t]=lt(V),y[t].c(),y[t].m(ce.parentNode,ce))}for(;t<y.length;t+=1)y[t].d(1);y.length=ve.length}if(_[0]&4){de=P(e[2].new_ctx.split("[SEP]"));let t;for(t=0;t<de.length;t+=1){const V=Fe(e,de,t);K[t]?K[t].p(V,_):(K[t]=nt(V),K[t].c(),K[t].m(fe.parentNode,fe))}for(;t<K.length;t+=1)K[t].d(1);K.length=de.length}},i:Be,o:Be,d(e){e&&(a(o),a(c),a(r),a(u),a(T),a(C),a(Q),a(R),a(I),a(w),a(j),a(Y),a(U),a(ne),a(X),a(ge),a($),a(me),a(ae),a(se),a(ke),a(ee),a(Ce),a(he),a(ie),a(we),a(te),a(be),a(ce),a(oe),a(Ee),a(le),a(xe),a(fe)),a(l),a(s),a(i),Z(M,e),Z(W,e),Z(F,e),Z(L,e),Z(O,e),Z(y,e),Z(K,e)}}}function ut(n,l,s){let i,o,c,{data:h}=l,r=0;const u=k=>{ct(`./${k}`),s(1,r=0)},f=k=>s(1,r=k);return n.$$set=k=>{"data"in k&&s(0,h=k.data)},n.$$.update=()=>{n.$$.dirty[0]&1&&s(3,i=h.all_queries),n.$$.dirty[0]&1&&s(4,o=h.query_data),n.$$.dirty[0]&18&&s(2,c=o.hops[r])},[h,r,c,i,o,u,f]}class mt extends ot{constructor(l){super(),at(this,l,ut,_t,it,{data:0},null,[-1,-1])}}export{mt as component,gt as universal};