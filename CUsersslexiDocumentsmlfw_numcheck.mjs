// Reproduce the per-statement SUM backprop and compare against a correct
// gradient computed with all reads taken at the start of the sample.

function tanh(x){return Math.tanh(x);}

function rng_factory(seed){
  let s=(seed>>>0)||1;
  return ()=>{s=(s*1664525+1013904223)>>>0;return s/0x100000000;};
}

// --- Reference: correct full-sample gradient (accumulate, then apply) ---
function trainRef(samples, dim, H, lr, epochs, init){
  let {W1,b1,W2,b2}=init;
  W1=W1.map(r=>r.slice()); b1=b1.slice(); W2=W2.slice();
  for(let e=0;e<epochs;e++){
    for(const sample of samples){
      // forward
      const acts=[];
      let pred=0;
      for(const x of sample.xs){
        const a=new Array(H);
        for(let h=0;h<H;h++){let acc=b1[h];for(let i=0;i<dim;i++)acc+=W1[h][i]*x[i];a[h]=tanh(acc);}
        let p=b2; for(let h=0;h<H;h++)p+=W2[h]*a[h];
        pred+=p; acts.push({x,a});
      }
      const dErr=pred-sample.y;
      // accumulate grads using a SNAPSHOT of W2 (correct)
      const gW1=W1.map(r=>r.map(()=>0)); const gb1=b1.map(()=>0);
      const gW2=W2.map(()=>0); let gb2=0;
      for(const {x,a} of acts){
        gb2+=dErr;
        for(let h=0;h<H;h++){
          const dA=dErr*W2[h];
          const dZ=dA*(1-a[h]*a[h]);
          gW2[h]+=dErr*a[h];
          for(let i=0;i<dim;i++) gW1[h][i]+=dZ*x[i];
          gb1[h]+=dZ;
        }
      }
      b2-=lr*gb2;
      for(let h=0;h<H;h++){W2[h]-=lr*gW2[h];b1[h]-=lr*gb1[h];for(let i=0;i<dim;i++)W1[h][i]-=lr*gW1[h][i];}
    }
  }
  return {W1,b1,W2,b2};
}

// --- As-implemented: in-place intra-sample updates (the code under review) ---
function trainImpl(samples, dim, H, lr, epochs, init){
  let {W1,b1,W2,b2}=init;
  W1=W1.map(r=>r.slice()); b1=b1.slice(); W2=W2.slice();
  for(let e=0;e<epochs;e++){
    for(const sample of samples){
      const acts=[];
      let pred=0;
      for(const x of sample.xs){
        const a=new Array(H);
        for(let h=0;h<H;h++){let acc=b1[h];for(let i=0;i<dim;i++)acc+=W1[h][i]*x[i];a[h]=tanh(acc);}
        let p=b2; for(let h=0;h<H;h++)p+=W2[h]*a[h];
        pred+=p; acts.push({x,a});
      }
      const dErr=pred-sample.y;
      for(const {x,a} of acts){
        b2-=lr*dErr;
        for(let h=0;h<H;h++){
          const dA=dErr*W2[h];
          const dZ=dA*(1-a[h]*a[h]);
          W2[h]-=lr*dErr*a[h];   // <-- W2[h] mutated here
          for(let i=0;i<dim;i++)W1[h][i]-=lr*dZ*x[i];
          b1[h]-=lr*dZ;
        }
      }
    }
  }
  return {W1,b1,W2,b2};
}

function predict(net, xs, dim, H){
  let sum=0;
  for(const x of xs){
    let p=net.b2;
    for(let h=0;h<H;h++){let acc=net.b1[h];for(let i=0;i<dim;i++)acc+=net.W1[h][i]*x[i];p+=net.W2[h]*tanh(acc);}
    sum+=p;
  }
  return sum;
}

// Build a multi-statement-per-program scenario
const dim=3,H=4,lr=0.05,epochs=200;
const rng=rng_factory(1);
const r1=1/Math.sqrt(dim), r2=1/Math.sqrt(H);
const W1=Array.from({length:H},()=>Array.from({length:dim},()=>(rng()*2-1)*r1));
const b1=new Array(H).fill(0);
const W2=Array.from({length:H},()=>(rng()*2-1)*r2);
const init={W1,b1,W2,b2:0};

// programs with 2-3 statements each (normalized-ish values)
const samples=[
  {xs:[[0.5,-0.2,1.0],[1.0,0.3,-0.5]], y:0.4},
  {xs:[[-1.0,0.7,0.2],[0.1,0.1,0.1],[0.9,-0.9,0.0]], y:-0.3},
  {xs:[[0.2,0.2,0.2]], y:0.1},
  {xs:[[1.0,1.0,1.0],[-1.0,-1.0,-1.0]], y:0.0},
];

const refNet=trainRef(samples,dim,H,lr,epochs,init);
const implNet=trainImpl(samples,dim,H,lr,epochs,init);

let maxDiff=0;
for(const s of samples){
  const pr=predict(refNet,s.xs,dim,H);
  const pi=predict(implNet,s.xs,dim,H);
  maxDiff=Math.max(maxDiff,Math.abs(pr-pi));
}
console.log('max prediction diff (impl vs correct accumulate):', maxDiff.toExponential(4));

// Also check single-statement programs: impl should equal ref exactly there
const single=[
  {xs:[[0.5,-0.2,1.0]],y:0.4},
  {xs:[[-1.0,0.7,0.2]],y:-0.3},
];
const refS=trainRef(single,dim,H,lr,epochs,init);
const implS=trainImpl(single,dim,H,lr,epochs,init);
let sDiff=0;
for(const s of single){sDiff=Math.max(sDiff,Math.abs(predict(refS,s.xs,dim,H)-predict(implS,s.xs,dim,H)));}
console.log('single-statement max diff (should be ~0):', sDiff.toExponential(4));
