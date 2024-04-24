gsap.to("#nav",{
    backgroundColor:"#000",
    duration:0.5,
    height:"100px",
    scrollTrigger:{
        trigger:"#nav",
        scroller:"body",
        start:"top -10%",
        end:"top -11%",
        scrub:1
    }
 
})

gsap.to("#main",{
    backgroundColor:"#000",
    scrollTrigger:{
        trigger:"#main",
        scroller:"body",
        start:"top -30%",
        end:"top -90%",
        scrub:2
    }
})

gsap.from("#pricing img , #dd",{
     y:50,
     opacity:0,
     duration:2,
     stagger:1,
     scrollTrigger:{
        trigger:"#pricing",
        scroller:"body",
        start:"top 60% ",
        end:"top 50%",
        scrub:5
     }
})

gsap.from("#upper",{
    y:-70,
    x:-70,
    scrollTrigger:{
       trigger:"#upper",
       scroller:"body",
       start:"top 60% ",
       end:"top 50%",
       scrub:3
    }
})

gsap.from("#down",{
    y:70,
    x:70,
    scrollTrigger:{
       trigger:"#upper",
       scroller:"body",
       start:"top 60% ",
       end:"top 50%",
       scrub:3
    }
})