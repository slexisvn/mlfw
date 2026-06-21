import { writeFileSync } from 'node:fs';

// Intent-structured chatbot dataset generator.
// Each intent = one canonical answer + many natural question paraphrases.
// We hold out a per-intent slice of paraphrases as a TEST set, so the test
// contains phrasings the model never saw in training -> a real generalization probe.

const INTENTS = [
  { a: 'hello how can i help you today', q: ['hi', 'hello', 'hey', 'heya', 'hi there', 'hello there', 'good morning', 'good afternoon', 'good evening', 'hey there', 'greetings', 'howdy'] },
  { a: 'goodbye have a great day', q: ['bye', 'goodbye', 'see you', 'see you later', 'see ya', 'take care', 'talk to you later', 'i have to go', 'gotta go', 'farewell', 'catch you later', 'bye bye'] },
  { a: 'you are welcome anytime', q: ['thanks', 'thank you', 'thanks a lot', 'thank you so much', 'many thanks', 'thanks very much', 'appreciate it', 'i appreciate it', 'cheers', 'thank you kindly', 'much appreciated', 'thanks so much'] },
  { a: 'i am a virtual assistant here to help', q: ['what is your name', 'who are you', 'what are you', 'tell me your name', 'whats your name', 'do you have a name', 'introduce yourself', 'who am i talking to', 'what should i call you', 'are you a bot', 'are you human', 'are you a robot'] },
  { a: 'i can answer questions about our hours location pricing and orders', q: ['what can you do', 'how can you help', 'what do you do', 'what are your capabilities', 'how can you help me', 'what can you help with', 'what are you able to do', 'tell me what you can do', 'what services do you offer', 'how do you work', 'what is your purpose', 'what can i ask you'] },
  { a: 'we are open from nine am to six pm monday to friday', q: ['what are your hours', 'when are you open', 'what time do you open', 'what time do you close', 'are you open now', 'what are your business hours', 'when do you close', 'what are the opening hours', 'tell me your hours', 'how late are you open', 'are you open today', 'what hours do you keep'] },
  { a: 'we are located downtown at fifth avenue', q: ['where are you', 'where are you located', 'what is your address', 'where is your store', 'how do i find you', 'where can i find you', 'whats your location', 'where is your office', 'give me your address', 'where are you based', 'what is your location', 'how do i get to you'] },
  { a: 'you can reach us by phone at five five five one two three four', q: ['what is your phone number', 'how do i call you', 'whats your number', 'can i call you', 'do you have a phone number', 'how can i reach you by phone', 'give me your phone number', 'what number can i call', 'phone number please', 'how do i phone you', 'what is your contact number', 'tell me your phone number'] },
  { a: 'you can email us at support at example dot com', q: ['what is your email', 'how do i email you', 'whats your email address', 'can i email you', 'do you have an email', 'how can i reach you by email', 'give me your email', 'what email should i use', 'email address please', 'how do i contact you by email', 'what is your contact email', 'where do i send an email'] },
  { a: 'prices start at nineteen dollars per month', q: ['how much does it cost', 'what are your prices', 'what is the price', 'how much is it', 'what does it cost', 'tell me the pricing', 'how much do you charge', 'what is the cost', 'whats the price', 'how expensive is it', 'what are the rates', 'how much per month'] },
  { a: 'we offer a thirty day refund on all orders', q: ['what is your refund policy', 'can i get a refund', 'do you offer refunds', 'how do refunds work', 'can i return my order', 'what is your return policy', 'how do i get my money back', 'are refunds allowed', 'is there a refund', 'do you give refunds', 'how long do i have to return', 'whats the return window'] },
  { a: 'standard shipping takes three to five business days', q: ['how long does shipping take', 'when will my order arrive', 'how fast is shipping', 'what are the shipping times', 'how long until delivery', 'when does it ship', 'how long for delivery', 'whats the delivery time', 'how soon will it arrive', 'how quick is shipping', 'how many days for shipping', 'when will it be delivered'] },
  { a: 'you can track your order from your account page', q: ['how do i track my order', 'where is my order', 'can i track my package', 'how do i check my order status', 'whats my order status', 'how do i see where my package is', 'track my order', 'is my order shipped', 'where is my package', 'can i see my delivery status', 'how do i find my order', 'how do i know when it ships'] },
  { a: 'you can reset your password from the login page', q: ['i forgot my password', 'how do i reset my password', 'i cant log in', 'how do i change my password', 'i lost my password', 'help me reset my password', 'cant access my account', 'how do i recover my password', 'i need to reset my password', 'forgot my login', 'reset password please', 'how do i get back into my account'] },
  { a: 'we accept all major credit cards and paypal', q: ['what payment methods do you accept', 'how can i pay', 'do you take credit cards', 'can i pay with paypal', 'what forms of payment do you take', 'how do i pay', 'do you accept visa', 'what cards do you accept', 'can i use a credit card', 'what are the payment options', 'do you take paypal', 'which payments do you support'] },
  { a: 'you can cancel anytime from your account settings', q: ['how do i cancel', 'can i cancel my subscription', 'how do i cancel my order', 'i want to cancel', 'how do i unsubscribe', 'can i cancel anytime', 'how do i end my subscription', 'cancel my account', 'how do i stop my subscription', 'i would like to cancel', 'how can i cancel', 'where do i cancel'] },
  { a: 'i am doing great thank you for asking', q: ['how are you', 'how are you doing', 'how is it going', 'how do you feel', 'are you ok', 'how have you been', 'whats up', 'how are things', 'you doing well', 'how is your day', 'hows everything', 'how you doing'] },
  { a: 'i am here and ready to help you', q: ['are you there', 'are you still there', 'hello are you there', 'you there', 'is anyone there', 'are you online', 'are you available', 'can you hear me', 'are you listening', 'still with me', 'you around', 'anyone home'] },
  { a: 'i am sorry i did not understand can you rephrase', q: ['what', 'huh', 'i dont understand', 'that makes no sense', 'come again', 'what do you mean', 'can you repeat that', 'i am confused', 'that is not clear', 'pardon', 'say that again', 'i didnt get that'] },
  { a: 'sure i can help you with that', q: ['can you help me', 'i need help', 'help', 'i need some help', 'can you assist me', 'i need assistance', 'could you help me', 'please help me', 'i have a problem', 'can you support me', 'i need support', 'help me please'] },
  { a: 'we have a wide range of products for every need', q: ['what products do you sell', 'what do you sell', 'what do you offer', 'what can i buy', 'show me your products', 'what is available', 'what products do you have', 'what items do you sell', 'tell me about your products', 'what is in your catalog', 'what kind of products', 'what do you have for sale'] },
  { a: 'yes we offer free shipping on orders over fifty dollars', q: ['do you offer free shipping', 'is shipping free', 'do you have free delivery', 'is there free shipping', 'can i get free shipping', 'do you ship for free', 'is delivery free', 'whats the shipping cost', 'do i pay for shipping', 'how much is shipping', 'is there a shipping fee', 'do you charge for delivery'] },
  { a: 'yes our service is available worldwide', q: ['do you ship internationally', 'do you ship to my country', 'is it available in my country', 'do you deliver internationally', 'can i order from abroad', 'do you ship worldwide', 'is it available globally', 'do you serve other countries', 'can you ship overseas', 'do you deliver to other countries', 'is international shipping available', 'do you ship outside the country'] },
  { a: 'you can create an account from the sign up page', q: ['how do i sign up', 'how do i create an account', 'how do i register', 'how do i make an account', 'how do i join', 'where do i sign up', 'how do i get started', 'how do i open an account', 'i want to register', 'how do i set up an account', 'how to create an account', 'how do i become a member'] },
  { a: 'our support team replies within one business day', q: ['how long for a reply', 'when will support respond', 'how long does support take', 'how fast do you reply', 'when will i hear back', 'how soon will you respond', 'how long until i get an answer', 'whats the response time', 'how quickly do you reply', 'when can i expect a response', 'how long for support to answer', 'how long until someone replies'] },
  { a: 'i am sorry to hear that let me help fix it', q: ['this is broken', 'it is not working', 'i have an issue', 'something went wrong', 'i am having trouble', 'it stopped working', 'there is a problem', 'this does not work', 'i found a bug', 'it is broken', 'i have a complaint', 'something is wrong'] },
  { a: 'that is great to hear', q: ['this is great', 'i love it', 'this is amazing', 'you are awesome', 'this is perfect', 'excellent work', 'i am happy', 'that is wonderful', 'this is fantastic', 'good job', 'well done', 'i am pleased'] },
  { a: 'i can only help with questions about our service', q: ['what is the weather', 'tell me a joke', 'what time is it', 'who won the game', 'what is the news', 'sing me a song', 'what is the meaning of life', 'do my homework', 'whats the weather like', 'tell me the news', 'what is two plus two', 'what day is it'] },
  { a: 'our products come with a one year warranty', q: ['do you offer a warranty', 'is there a warranty', 'how long is the warranty', 'whats your warranty policy', 'are products under warranty', 'do you have a guarantee', 'is it guaranteed', 'how long is the guarantee', 'does it come with a warranty', 'what does the warranty cover', 'is there a guarantee', 'tell me about the warranty'] },
  { a: 'you can change your plan from billing settings', q: ['how do i upgrade', 'how do i change my plan', 'how do i downgrade', 'can i switch plans', 'how do i change my subscription', 'how do i upgrade my plan', 'where do i change my plan', 'can i upgrade', 'how do i switch my plan', 'how do i modify my subscription', 'how do i change my package', 'how do i pick a different plan'] },
  { a: 'yes there is a free trial for fourteen days', q: ['is there a free trial', 'do you offer a trial', 'can i try it for free', 'is there a trial period', 'do you have a free trial', 'can i test it first', 'is there a demo', 'do i get a trial', 'how long is the free trial', 'can i try before i buy', 'is a trial available', 'do you have a demo'] },
  { a: 'yes your data is encrypted and secure', q: ['is my data safe', 'is it secure', 'how do you protect my data', 'is my information private', 'do you keep my data secure', 'is my data encrypted', 'how safe is my information', 'do you protect privacy', 'is my account secure', 'how do you handle my data', 'is my payment secure', 'can i trust you with my data'] },
  { a: 'please describe the issue and we will look into it', q: ['my order is wrong', 'i got the wrong item', 'my package is damaged', 'item arrived broken', 'my order is missing', 'i received the wrong product', 'my delivery is late', 'i never got my order', 'my item is defective', 'the product is damaged', 'my order never arrived', 'wrong item in my order'] },
  { a: 'sounds good let me know if you need anything else', q: ['ok', 'okay', 'alright', 'sure', 'got it', 'sounds good', 'understood', 'fine', 'cool', 'great', 'no problem', 'makes sense'] },
  { a: 'no problem feel free to ask anything', q: ['can i ask a question', 'i have a question', 'may i ask something', 'quick question', 'i want to ask something', 'can i ask you something', 'i need to ask', 'got a question', 'mind if i ask', 'one more question', 'another question', 'let me ask you'] },
];

const TRAILERS = ['', '', '', ' please', ' thanks'];
const SYN = [['hi', 'hey'], ['hello', 'hi'], ['whats', 'what is'], ['cant', 'can not'], ['dont', 'do not'], ['i would', 'id'], ['how do i', 'how can i']];

function augment(q) {
  const out = new Set([q]);
  for (const [a, b] of SYN) {
    if (q.includes(a)) out.add(q.split(a).join(b));
  }
  const base = [...out];
  for (const phrase of base) {
    for (const tr of TRAILERS) if (tr) out.add(phrase + tr);
  }
  return [...out];
}

// Deterministic shuffle (seeded LCG) so the split is reproducible.
function shuffled(arr, seed) {
  const a = arr.slice();
  let s = seed >>> 0;
  for (let i = a.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const trainRows = [];
const testRows = [];
let intentIdx = 0;
for (const intent of INTENTS) {
  const variants = new Set();
  for (const q of intent.q) for (const v of augment(q)) variants.add(v.trim());
  const shuffledVariants = shuffled([...variants], 7919 + intentIdx * 131);
  const nTest = Math.max(2, Math.round(shuffledVariants.length * 0.25));
  shuffledVariants.forEach((q, i) => {
    (i < nTest ? testRows : trainRows).push([q, intent.a]);
  });
  intentIdx++;
}

const trainShuffled = shuffled(trainRows, 1234567);

function toCsv(rows) {
  return 'question,answer\n' + rows.map(([q, a]) => `${q},${a}`).join('\n') + '\n';
}

writeFileSync('examples/chat_train.csv', toCsv(trainShuffled));
writeFileSync('examples/chat_test.csv', toCsv(testRows));

console.log(`intents: ${INTENTS.length}`);
console.log(`train pairs: ${trainShuffled.length} -> examples/chat_train.csv`);
console.log(`test pairs:  ${testRows.length} -> examples/chat_test.csv`);
