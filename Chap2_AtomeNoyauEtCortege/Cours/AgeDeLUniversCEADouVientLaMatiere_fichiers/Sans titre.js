(function(c,d,a,f,g){if(c[d]){var h=function(){if(a.gifPostbackUrl)return a.gifPostbackUrl;f.error("Cannot retrieve click ID. Please, check your offer set up.")};g=c[d].state.callbackQueue;c[d]=function(b){b(a.clickId)};var e=new Date;e.setTime(e.getTime()+31536E6);document.cookie="vl-cid="+a.clickId+";samesite=Strict; expires="+e.toGMTString()+"; path=/";c[d].registerConversion=function(b){f.warn("Deprecated. You can now use getClickID method to retrieve Click ID and logConversion method to fire conversion postback");
b(a.clickId)};c[d].getClickID=function(){return a.clickId};c[d].getTokens=function(){return a.offerUrlParameters||{}};c[d].getConversionPostbackPixelURL=function(){return h()};c[d].logConversion=function(b,c,d){var a=h();a&&(b&&(a+="&payout="+b),c&&(a+="&txid="+c),d&&(a+="&et="+d),b=document.createElement("img"),b.src=a,b.width=1,b.height=1,document.body.appendChild(b))};g.forEach(function(b){return b(a.clickId)})}})(window,"dtpCallback",{"clickId":"ws9nsu36pj2u5fmriai8td96","campaignId":"3756d6de-b658-4c73-803b-454959709b18","gifPostbackUrl":"https://landing.qr-code-generator.com/conversion.gif?cid=ws9nsu36pj2u5fmriai8td96&caid=3756d6de-b658-4c73-803b-454959709b18&script=pCcmEXsP","offerUrlParameters":{}},console);