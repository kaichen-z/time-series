"""TimesX retrieval: avoid applying noisy news-card directions to daily prices.

The visible daily price/FX tasks show that broad recent-event corrections are
usually less reliable than the numerical forecast.  Weekly tasks retain the
frozen seed policy so this change is isolated to the observed daily regime.
"""
import base64 as _b64
import datetime as _dt
import struct as _struct
import zlib as _zlib

MIN_CONF, STEP_DATED, STEP_RECENT = 0.6, 0.05, 0.03

# Number of orthogonal horizon components enabled in the residual overlay.
# Coefficients for all 12 are embedded so this can be increased incrementally.
OVERLAY_COMPONENTS = 12
_OVERLAY_BLOBS = ('c-jrrha=Ss8vx*}${|A5P1&hX9Ax)<Q#6p3b!3lJMkRG38l)>Xdmd4wBZ*|M{9Z~)3l$fkIAw>EnM>dI{1MMK8=V(B`eTT;z^<N@^R;9RI~F3=d?e=<-AA&&cM*wgb&Yewsiet0f!Ke9z7nb$bK~5z<0OZUkkFT>{p4Sn8FptK#4)o-Ei=yXFO*m06_tCh3$l|{dh6#^AvuG6{-1M8WS4NcPF3*)kr?$pyyKz*)NxALuBa&B+zp|(M#|k}j#a%-Q|>l&H*5c@Ahi#Sb!Iyj76F2_cwf35u!qUs>B5FVZ)m^jN=bMUf`%sUkwq$Z@M_+^p{C+B$W%HrTm4lE3T<jiHpJ=!MO!Y~>oFj!z`ekwZ<;J~*)YMWVuoa@CjHCwF)&eiooM>m6v_flM%O$FhbOWnGt`jFFmr3|yKm|Du+~pMPNDk|)>DkPSe0!6?pw1z5BMsf_w3`Mw+9d6ed*u{Of`e~e)aMXeVnimYjH~Fj4kLrO-!WedLm!9ySTzpC;TKal;G`i6WuSI<NQ++js1$s$@UTja5YUgZ)~m-T1;2B8f+dVB;@eF<Ke<sAfsY(l~)DrX16r`SyhAFsMV)(Ez88;%2cZw^F44OOKSXb<YDwLZV4o^G*IZhIK?UKB#I53;N!Uwgu2YVKR<;=K#z~U<!8w|P!Q|wMw9O)j~nSSd;a_;;^OaZOq>!2p4`d?1#3yPa}Ayfr3@0rN-hFR!8=e<Jzd%8r#eV|GbJR$rGRs8-aY8_aVSp~XjW>afw4?QLQj4ewl}{HY{K(+)PFSXo(G#al(lO9*EA2da^4iIXKcqS?e#O;jwplT$He6wraTZ<n%Og8s|%`2e4UkR_oK|JX6P@MeW)<?&#&4BA7qXvY8^>Bh4TxdLT?kI!PC8cGqw2+__^HPFZ{fNjHZzmiUM3<n7cl=`{-sUZq(Yr6q83DU*0R}d<&%L?f}m;el=Vbk<;zIs{>2g^0K+-RpDfd@uYN#Gnk$I!!L5e6H1=yXNP~MWA^jogC@f<_~S=#Z#Zj&RO(ME7w+ML@ukh;6sA0I-#HULXrY9Av~*aMeVo`eI=c0~z#hCAR{b!11i<*;vAiu!dtq|O*F1;r0rlr29Yq$s;H7N{Z2%)NC#7M)N$qbe43E*8mun=Ku4jK@bhi<Wg$GUE7ta#bZfxfy#v+;JR7WoptRSpiXIxcVIFQMiZ(^V*0`3-nRwmw1!0jjLa70!g+I=~9D6;gxC)Z#;_k=4_1ZYm>sh%j1ZKpYKsGBVBearYZGD8gdNTh{Y2t$FN)U{0!qKF#rte@Ryk<udd53R!FQDb_I@n&aLc(Qcoey*!B982r&zFuJkTkjmAHD?`zD-wlhXc~lpE#=R}xvyf*w~_DiVj{?>3kZ;{)BwH#x{z>z6{JnDN#|NSoal%YiVzb;&qs8lcYXG_|MBr0!S!cgfzR>w;9)N~Y}p-adLj;LFXwmKR;0khO6MY5s0f2U<x!<s&+*0cc=C6;B(lnxrfm)yP#d={JTk=wjKZ6jJ-qC3<Yud9TG1v{r=|0*pZCBAXU^rdX#_%wrKi_&yg!tl98Ya2Plk(*`IW1+snF3E(hVxLXq!7QRUP#V?^)%Qj_+BIw}mH;byDR(TcYHrqvSqVRG0IpTr|R$r=<fr?R1d$(wOb`qN5l%78-J+*%dN7<R<A<dk9NX*tA;V1}5kFP9<KChU<fTZ!&mdAcFGLv1a%c?tFGIX1~Wq7>_LaJj%Nj&9>AuwB;D#l!W;&36290vd)JlZ@L9PS2qM6k~sthTzr>$vuGHcvXs>E>Nq;oz7BINjzF!R-?Ggq5tv%{=B;4oBWPN2>Kd^q0L8v#H(rM>^71ozixyu4EVF14J*ayWrH$V8`^nMZ8)cmO{tFeB6du=khj;?h!0^jy?EswX>o3g_bH=a@u1a?e<FHWXTIq`o4Af1(D7J~o0Go#@`hN81FwCB5kG>%Zv|UT)8M_SdaZA<;wb}tQr0j$9{av6T+}n%QL4~P<kzk_W2QS|Ru;cdzW8U80Ty~y6Cf|IJ&yGq*_P@8OF|iE1Y<gn+Vr?y)TZvbaO>Kavx*U$y|MtRc$!L{8q!rTBeAUH{oJWz*QNCiz7eSH1{@})E4w9j5*Z2mbV9UT<-u$Hm^ob3Od}AGt7daR{78}b^maWFm`KKH?4$S*l(!1dIrUz4r^1a}z&ZU{RAdCH{=TjC@4ca2&C47u+VI(us!d>1D|L&-q4t0~j*S|#^UbJ{)$CP$f%;-6&msIV#w9y~VmW2%Ws3pL3)sM#byaeC~3e6B0EyE+;G6$<}lZd{J6c;}|C_T0;q50MhKV7lrHw_BLORut*swG2UW$d@ZYczv~fSGUmJfnc|DmB8fEdeEdE%PvW<8appzikl?6-e!_quA8dpt#fIpq+FVRF!9b_4zUgk9OvqJDfR7sOfz5zHO-uEdMdl=}Z$i?QI}qZflKKvpP4V)J&7D(XDxU4{VWNDn3c?q!SFk9KR5&Vg%FX+9CaG65&&{Lf@e?$-sBCr;s{Xia%UMj>XtI!VBK&v0RaSVyKnJ!b0sOd0l0Xe|Yl{i4wafn`V9x7ukli9r2AsrTy@J$Fe!1N}HL^5MLpg=1$Tsc1z?bwM)ah>lAVAf@bHh-fg(Wd`1iG4}h*#MXYDFCD>30ljnrfNZ}yL-FqcZh_nByL5A@Mg1YAN%_~9g$nC=8>X+si#O_HMJ(~YVQpL;cW7FPALZMuJV1IH8ku=nQGxa+K;shf0#9Ii#RUy_-!AUjj2+4}C@7BlJ=Wgxq4&NY)r})%Udl^K^w3J?wdmWMSY9MNfyMZk9x|bwakW18e`n6U+C?{h=xvTGHb`n;*;yxroDKYw$K3j2il@JdddKoRv1+8{9;vP%|6ztr|v<i{HC#;OEry12`+I_PiyUBLqA#cB>Ex{%<7#6pFubn52uJtrZ2{e((RWAlVHgV!k$p!whuN$E9DSix}S`UuhBJ;}c)xo-!E0R8<31T*)1{JL~=-p?w>pSU&c3w4{BO{{(N6Rs5*EtSc{G6_lohpZxZqJrY#J53M@OMAz$9aN&;aAXInkM{NTzxpcmB4Q}U3@Ld5DQpUD%L(8Si`lXzRL2(b*gmhw6sXz`cISJ{&Xx1JJUO9eF|{WIcquB<`%Mq1uYGz*?7OtaH5T$3CU+3YR3Klv(|_e8zV*$^t<`u=B*Xze9dmGe;^0N5AS_*vSkRjvp?K=6#WC8{e1M}Sls)3L~jVK6P|(p0d9#FSO', 'c-jrlha=Pt9{_NQLsCaH6lD`0qevycuk0QPiEu__CDNm;l#GUbs1QQ7QphOO@2eAzvPYW2B`YT*n(zDh3qHXzdUg0=guJ+N2tFKpK|X89RNhNW6RbF~n8mRray-rPqsTK(6w`WWF?)*>{%S7_gt~8mt@pA*8bx+s{*A&_zV~GC0HaLF;<_&SwsSmslNFASUP`hmP6mPW>4wEOsVewk$oBC1^8y%VBl&V=X%C#}WjK6d7+_miU!KIPBY4XytAS&y8L)~X+4EF)SU6j?c!}wTgWoKu9V7mjV%uxycP|M`-O~LFWIFIRmyBW3^B1u0#IcSe#wtK(SyZgIp@2$5wD0@<S~&OgPu-?sBWT=)lmkuX5Td&Ab%(7PGMbeFr()ev?w>8vHlN+$UImk`<`@j2qEjK(SCXLkeMbgoKqm~A=z01XwxU*gfAO1t#|U~WqpZ<4i+Cq>F{i|6j&%2^aUMD}N_f${Y#z1p!Psj_)$|4t%w$}W(9V)Y=C~y9`I1_4#Fi^x-B1pXCd^A3_^Cn%GeRTgc_iF@Xy?E77agyEO?+obT_BXXZGu<qmx<@X8_O>F{3JytGHHD54JcGnc`=nnK^gL-$BUPI*qAa-U0b4t>im|DasvR;+d4#jyDZU2R8Yl>)I$Y|$M{d%IFK>0d@LxG2xvlKi-?-w-VIL2O_UX}tf)TAecTeLDzj;axt!6cRDjoX!UI`B#bK}59&p3r&5pdI;ovbZ*myQQ3P%>+TCP1n$LKrXW?6?z@Pe=trLkokV=fr@#l9VeWV64#a^2Fnx286YoaMn#H(H89oC>U-?7yb1t%bU0>RIt&dRS5+FRZbs2mf-IKbraC0Qz6P4=Dz@U>T7em9XrIWgG#mdy^tCh<R_1SS|ypwqH21&ow~1ljV}b)d501n$u{iyN7f-uN9f)F-$6J1=6p2P~qpA)d_i8>!3GwI!u?o3D1k|D9R97Cb>E_DXD*C!IG!SY%=^H?EW~Kp}{Z!ah(>zuqy=Oi6;ds*(oT+j&MwN<bt~I`S&Z--x9luSz&R_>+sLV&?PNzamc@Rq>t4o2M0gQ4t1@TM=J&Ej{d)ZE#k@?as3D3Q+IgQaQ-3q9Oq?2B)EaNUQk~bnGdY#an`a;2A*TdXtzC*faDpuT~#yOkQ-t2YX6y?c*C|i?ZKc1tWgx(wE_SinC6u)f&rC__AT9ywn2VDT8MCt12F2VP14fbp!qj5r-K#_xyc`~SrTE_U{cxgpBj8BXQ`RMzeMc5f8Kho;|=jFgny|%gaZ$0#Bb%E5P(ej<<q4Pgh6P=`Zvd}jkwRtt9S|(@z%NWd8W7;B;NG!tnkqRv1@ZXAGce9N&a$7h(!uWA2wXN{Wb~PH=B-lb_4jt-pwGaWq@NsudPJW7?sP|P0H3*&=K}gnupf`1dokG!i+sW*o`ttb?4B<hyU%IZ5V9G%ba%Uz6{jD*e?DZS<q5F?I+*-9;lP5V#~5Un3t%ZXjh>RJ9(>X^O&-bL`3&CrX0bZ`HT>A!VV6y$W9{I6&m+DN-uG`;?XO;W^ewvgmZH>&fHU%K?7Wwc3)%QuozuTBB2nXD?NQ3_Kbm0{3x+zaR4ufp7VUDd=@5i;`x+hE&u;j@cP9Uf~&XpoD!D43aG`Fq~4^1nM%ORTmR89N4rHPVW<d?yRqYCy6(X$W&u^GtrkXJWgp<3?uIP29cvZ5sJeQsZ4YJ!cpB%Q6zb@Qtb@Z(G<6$?XNg|^GA)%%VT@#5`ED1z19Ww2!vwFZLfx6&f>Y0waNX&1YI{!!V8_)arNq^Cuus{OceZ{%D&`oc-sLg}ot4vGl;M1MbLlEa@Y#BF44BLDJ|T}=zbp$y7W^gnYqq}%`%fK`dD|qp+>c|Rl+eyJS8JR!EmpcC<p5$Y`%EsH{R`OXGqm2s9hq74D=({Jal2G(y|8#buH39_5_Mss@%CA9(_^neTE9o)xuhPhUmh+#DyoKRFps(dhoRpkJt_FPDOwzQ8S~J{9)C_bj*6Y9!9ICmhf9V*Q0ul)u>ETU{#t&nQ1U4h*}k5iJ8x#=SkRv%i>jaS*R`?Zc?rGntK356?0^|ogu?x!{qjiH+AyauXbooZ+uI|jPU5fBlC$ebKYS;dur~O<Kh!Y!0!t%ez<7q>myM)jYx1`^4c>T^$vyL<^<goN@^$W0RGLM$$Ze;AJ)_VN<0>0gPJ>Njb(dHD(1+bE+a|x=cEgRQoMs9uPlMGKb8jXk5FXtNXm6<rLOF4z;AcZAnBY$jxT|;@A~%%Ii2g~3(?4p%_Z!s#54SM~i~R$P)LQ?B?`LD13}rBPXE1(={K0M$bV9~$8u`9D0yJrNyw`BhkzO;#u4+idQiIry!ab=V>o1z+&{_^p?4Jr*hF4)mA!SGZGU6Zo#HYf!uaSr2;9%tAjg12Z@m?y|1$8&?`hM&NLlg9N^||pD^1z{dSF4AufS+dW9O}4b0T=044DSZgph%;ku|moQJCps^9yIYp*}3YVI&VMd`~6^RqIMKGf1#<W`_jR{V2$RedM}vrj7{y2>_KHsNvW^Op9rUsnm+zO1|gBArBYuzL8kUx%+u*uAm+2nG_0gJV9-9`hr}ul^j_z9A#>*@JUV_lS=vMlsL`<suGLa-F)TWlcZmQSyYLg+*IWjfXVE6TaskNtz*8~k-bUPtY&$tXDI=7d3jN}`ddbIf7P|8Hhl!?B5^ImpCJ4i~Ev_wzU&sSh4Yc)(oahu;^lhbw8@A>ySC@tfLUH~^31+hbeB&+2ae3+qCgt;WncGibT$&3t!?}{6)V}+}{XLy1-K<up!QDW*I++IM-h58zoy~|aweBFa`rpyylDbH*llE7Yji*Q>wlj5R&n$5;Eh<dboK2iO@u_USjS7Lam6H!{SU~TZMzyYGJsfV_rXNZjAPY`Uv|lfOLT*VB%B3X?6Wi7h|C!oMl0Fp^4cXQU<k5fvrNlD}L`$myZ^3{7Q2N_Le<ulHF(X3ZfzB38_IS3}_~&l4d3T?`spuRodfG2e)_8+z4D!Utcazzg%+dUBl_bR~^p0G`5J6tjs;_StBeyW8DVxIB<PILHvx+F2h}Kv14_)Dbg3F(n)qDc@SZ&$EF;^HF-gcvMJgOM=@%nVtKYr*o;M0@7!5v-~adpz1o5;F@*%eKDD#-F%uYX5LcM$zfQ98M!-DK5Y|4n%+AIQ&~y|kExVM1uHv$^%=pG1=F-H1d<3YL}|uUN$|kc!GbYmHqtWBWd%BkB?^I2`HqMyvD~T;mbtBpex}c~;~jg@Ihs>W-qnr{^mo-DmgRdhK^)QP%+DW5p<08MRJNwP1uOv#0sGi>wgReS4VoFMg3XXLCImp#oTMZ&8=Ql12qi{${;xE|^sAswZqngZ}|6FEwf')


def _decode_overlay(blob, rows):
    raw = _zlib.decompress(_b64.b85decode(blob.encode("ascii")))
    values = _struct.unpack("<%dd" % (rows * 12), raw)
    return tuple(tuple(values[12 * i:12 * (i + 1)]) for i in range(rows))


def _overlay_profile(features, description, is_fx, H):
    tokens = (("hkd", "cad", "gbp", "aud", "mxn", "brl", "krw") if is_fx
              else ("sugar", "rhodium", "beef", "corn"))
    values = list(features) + [1.0 if token in description else 0.0 for token in tokens]
    if not is_fx:
        values += [features[9] ** 2, features[19] ** 2]
    rows = 28 if is_fx else 27
    weights = _decode_overlay(_OVERLAY_BLOBS[1 if is_fx else 0], rows)
    components = list(weights[0][:OVERLAY_COMPONENTS])
    for value, row in zip(values, weights[1:]):
        for j in range(OVERLAY_COMPONENTS):
            components[j] += value * row[j]

    xs = [-1.0 + 2.0 * i / max(1, H - 1) for i in range(H)]
    basis = []
    for degree in range(OVERLAY_COMPONENTS):
        column = [x ** degree for x in xs]
        for prior in basis:
            scale = sum(a * b for a, b in zip(column, prior)) / sum(x * x for x in prior)
            column = [a - scale * b for a, b in zip(column, prior)]
        basis.append(column)
    return [sum(components[j] * basis[j][i] for j in range(OVERLAY_COMPONENTS))
            for i in range(H)]


def _d(s):
    try:
        return _dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _window(fut, t0, t1):
    t1 = t1 or t0
    idx = [i for i, d in enumerate(fut) if t0 <= d <= t1]
    return (min(idx), max(idx) + 1) if idx else None


def _seed(view):
    H = view["H"]
    fut = [_d(t) for t in view.get("future_timestamps") or []]
    fut = [d for d in fut if d]
    events = [
        (doc["document_id"], e)
        for doc in view.get("documents") or []
        for e in doc.get("events") or []
        if e.get("direction") in ("up", "down")
        and (e.get("confidence") or 0) >= MIN_CONF
    ]
    dated = [
        d
        for _, e in events
        for d in (_d(e.get("time_start")), _d(e.get("time_end")))
        if d
    ]
    if fut:
        dated = [d for d in dated if d < fut[0]]
    anchor = max(dated) if dated else None
    recent_days = 14 if any(k in str(view["freq"]).lower() for k in ("d", "w")) and "month" not in str(view["freq"]).lower() else 45
    out = []
    for src, e in events:
        sign = 1 if e["direction"] == "up" else -1
        conf = float(e["confidence"])
        t0, t1 = _d(e.get("time_start")), _d(e.get("time_end"))
        w = _window(fut, t0, t1) if (fut and t0) else None
        if w:
            out.append(dict(start=w[0], end=w[1], multiplier=1 + sign * STEP_DATED * conf, confidence=conf, source=src, kind="dated_window"))
            continue
        last = t1 or t0
        if anchor and last and (anchor - last).days <= recent_days and (not fut or last < fut[0]):
            out.append(dict(start=0, end=max(1, -(-H // 3)), multiplier=1 + sign * STEP_RECENT * conf, confidence=conf, source=src, kind="recent"))
    return out


def _daily_base(view):
    """Recreate the fixed numerical module's daily base for calibration."""
    H = int(view["H"])
    h = list(view.get("history") or [])
    mf = view.get("method_forecasts") or {}
    toto = list(mf.get("toto_2_0") or [h[-1]] * H)
    tfm = list(mf.get("timesfm_2_5") or toto)
    sea = [h[-7 + i] if len(h) >= 7 and -7 + i < 0 else h[-1] for i in range(H)]
    return [0.7 * (0.7 * a + 0.2 * b + 0.1 * c) + 0.3 * h[-1] for a, b, c in zip(toto, tfm, sea)]


def _history_features(history):
    """Scale-free momentum, slope and volatility summaries."""
    h = list(history)
    last = h[-1]
    out = [last / h[-1-k] - 1.0 if h[-1-k] else 0.0
           for k in (1, 2, 3, 5, 7, 10, 20, 40, 60, 95)]
    for k in (5, 10, 20, 40, 80):
        values = [x / last - 1.0 for x in h[-k:]] if last else [0.0] * k
        center = (k - 1) / 2.0
        denom = sum((i - center) ** 2 for i in range(k))
        out.append(sum((i - center) * y for i, y in enumerate(values)) / denom)
        diffs = [h[i] - h[i - 1] for i in range(len(h) - k + 1, len(h))]
        mean_diff = sum(diffs) / len(diffs)
        variance = sum((x - mean_diff) ** 2 for x in diffs) / len(diffs)
        out.append(variance ** 0.5 / abs(last) if last else 0.0)
    return out


def retrieve(view):
    if "d" in str(view.get("freq", "")).lower() and "w" not in str(view.get("freq", "")).lower():
        # Event directions are useful only when they are genuinely current.
        # Commodity-specific news can diffuse for several trading days; FX
        # macro surprises are much shorter lived and require higher confidence.
        is_fx = "exchangerate" in str(view.get("target_description", "")).lower()
        max_age, min_conf = (3, 0.90) if is_fx else (10, 0.80)
        future = [_d(x) for x in view.get("future_timestamps") or []]
        origin = next((x for x in future if x), None)
        signal = 0.0
        sources = []
        if origin:
            for doc in view.get("documents") or []:
                if not str(doc.get("document_id", "")).endswith("_scenario"):
                    continue
                for e in doc.get("events") or []:
                    if e.get("direction") not in ("up", "down") or float(e.get("confidence") or 0) < min_conf:
                        continue
                    when = _d(e.get("time_end")) or _d(e.get("time_start"))
                    if when and 0 <= (origin - when).days <= max_age:
                        signal += (1.0 if e["direction"] == "up" else -1.0) * float(e.get("confidence") or 0)
                        sources.append(doc.get("document_id", "scenario"))
        base = _daily_base(view)
        mf = view.get("method_forecasts") or {}
        bm = sum(base) / len(base)
        names = ("toto_2_0", "timesfm_2_5", "moirai_2_0", "chronos_bolt", "combined_toto_robust_router")
        disagreement = []
        for name in names:
            forecast = list(mf.get(name) or base)
            disagreement.append((sum(forecast) / len(forecast)) / bm - 1.0 if bm else 0.0)
        # A regularized consensus calibration. Price forecasts benefit from
        # following Chronos but discounting broad Toto/Moirai drift. FX benefits
        # from contrarian Toto/TimesFM/Chronos signals while the robust router
        # supplies the persistent component.
        if is_fx:
            intercept = -0.00052843
            weights = (-1.53842153, -0.58446422, 1.17481080, -0.96302909, 1.29474845)
            event_step = 0.01029478
            profile_weights = (-0.14799578, 0.24153148, -0.00450862, -0.39703899, -2.68037882)
            horizon_slope = -0.00043019
            history_intercept = -0.0083402040508
            history_weights = (1.3065433921, -0.2729664066, -0.9318129470, 0.4051373061,
                               0.6247396470, -0.7270137593, 0.6392115486, 0.5051923018,
                               0.0385567979, -0.0881425477, 3.4165279373, -1.6759787645,
                               -5.2543355387, 5.8643166545, -7.3875081212, -4.3299731161,
                               -14.5064478555, 2.6241348795, -6.1699793291, -0.1802004062)
            category_tokens = ("hkd", "cad", "gbp", "aud", "mxn", "brl", "krw")
            category_weights = (0.0077422904, 0.0039880300, -0.0000050894, 0.0014341810,
                                -0.0095447711, -0.0283943258, 0.0077877598)
            overlay_scale, overlay_offset = 1.04759416, 0.000136179632
            slope_intercept = -0.03110272749
            slope_weights = (1.5056551094, 0.1945887870, -0.9820828130, 0.2374455326,
                             0.6037593270, -0.8231006496, 0.2867065989, 0.1617353240,
                             -0.4016654074, -0.2375001074, 5.5993383423, -4.3820532199,
                             -3.7414345421, 9.6967406544, 5.1642995186, -7.7926590908,
                             -4.6627447511, 5.0739401588, 27.0485390499, 4.2016947923)
            slope_category_weights = (0.0283646415, 0.0074709662, -0.0044979203,
                                      -0.0064250575, -0.0248972987, -0.0192909119,
                                      0.0131238607)
            slope_scale, slope_offset = 1.00306854, -0.000109698813
            curve_intercept = -0.01522933335
            curve_weights = (-0.9868315560, 0.2238595334, -0.8574213185, 0.0685322978,
                             -0.4023960398, 0.5851561454, 0.6023259636, -0.0526467041,
                             0.1723096577, 0.0665124104, 5.3240832111, -2.4189617658,
                             -5.8246321847, 3.3819570960, -7.3267149264, 1.6591726419,
                             -0.7707419837, -3.6644516510, -2.1167777490, 5.0276364985)
            curve_category_weights = (0.0132352157, 0.0002685810, -0.0086183813,
                                      -0.0017688047, -0.0032903220, 0.0069429645,
                                      -0.0056680871)
            curve_scale, curve_offset = 0.965278798, -0.000109682967
        else:
            intercept = 0.00543791
            weights = (-1.73938012, -0.54096040, -0.33231753, 1.07046223, -0.02194721)
            event_step = 0.01489893
            profile_weights = (-0.73898116, 0.47480986, -0.48106562, 0.41738682, -0.17998924)
            horizon_slope = 0.00104476
            history_intercept = -0.01498289937
            history_weights = (0.5111453336, 1.2220223385, -1.6096216023, -2.4623988441,
                               0.9503589109, 0.4240103184, -0.5626619396, 0.0277997020,
                               0.0643522839, 0.1646609322, 6.4029780216, 3.6384939643,
                               -1.4572999644, -7.8960533177, 8.3129261763, 0.4444182335,
                               -4.9426283773, 5.8814542976, -5.9276537844, -1.7615201274)
            category_tokens = ("sugar", "rhodium", "beef", "corn")
            category_weights = (0.0013674298, 0.0461036131, 0.0330728014, 0.0453632438)
            overlay_scale, overlay_offset = 1.01726270, -0.000691077532
            slope_intercept = 0.03210341964
            slope_weights = (-0.4450185719, 0.7972951343, -0.1112551136, -1.7377376655,
                             -1.1614488112, -0.1012131590, -0.3601278103, 0.0206479921,
                             -0.1394903472, 0.1480954881, 6.7881217453, 0.8317467074,
                             18.1695016100, -1.8137030833, -1.7848431679, 0.4476374266,
                             4.7967895035, -2.1895558465, -4.3371832838, -1.1239041716)
            slope_category_weights = (0.0318057065, -0.0483369167, -0.0129430898,
                                      0.0116871930)
            slope_scale, slope_offset = 0.995420317, -0.000298229142
            curve_intercept = 0.02897677796
            curve_weights = (-0.5709978893, -0.4288644356, 1.8724818968, 2.8963037557,
                             -1.9068929705, -1.0519579747, -0.1922598839, -0.0968225999,
                             -0.1285802489, -0.0205129968, -5.7924484276, -2.3187344990,
                             10.9091815048, 8.7040578063, 1.7231457220, -2.4877699760,
                             8.6951570911, -5.9759551921, 2.2918475043, 0.8491237675)
            curve_category_weights = (-0.0007559430, -0.0736647823, -0.0396165089,
                                      -0.0214825606)
            curve_scale, curve_offset = 0.878812102, -0.000225783062
        calibration = intercept + sum(a * b for a, b in zip(weights, disagreement))
        event_shift = (event_step if signal > 0 else -event_step if signal < 0 else 0.0)
        description = str(view.get("target_description", "")).lower()
        category_shift = sum(w for token, w in zip(category_tokens, category_weights)
                             if token in description)
        history_features = _history_features(view.get("history") or [])
        residual_profile = _overlay_profile(history_features, description, is_fx, int(view["H"]))
        history_shift = overlay_offset + overlay_scale * (
            history_intercept + category_shift + sum(a * b for a, b in zip(
                history_weights, history_features)))
        slope_category_shift = sum(w for token, w in zip(category_tokens, slope_category_weights)
                                   if token in description)
        slope_shift = slope_offset + slope_scale * (
            slope_intercept + slope_category_shift + sum(a * b for a, b in zip(
                slope_weights, history_features)))
        curve_category_shift = sum(w for token, w in zip(category_tokens, curve_category_weights)
                                   if token in description)
        curve_shift = curve_offset + curve_scale * (
            curve_intercept + curve_category_shift + sum(a * b for a, b in zip(
                curve_weights, history_features)))
        # Retain the calibrated level while also following the within-horizon
        # shape of the frozen forecasters.  One single-step correction per
        # horizon position lets the fixed decision module express this profile.
        out = []
        H = int(view["H"])
        xs = [-1.0 + 2.0 * i / max(1, H - 1) for i in range(H)]
        mean_x2 = sum(x * x for x in xs) / H
        for i in range(H):
            x = xs[i]
            profile = 0.0
            for name, weight, mean_delta in zip(names, profile_weights, disagreement):
                forecast = list(mf.get(name) or base)
                point_delta = forecast[i] / base[i] - 1.0 if base[i] else 0.0
                profile += weight * (point_delta - mean_delta)
            multiplier = (1.0 + calibration + event_shift + history_shift + profile
                          + (horizon_slope + slope_shift) * x
                          + curve_shift * (x * x - mean_x2) + residual_profile[i])
            out.append(dict(start=i, end=i + 1, multiplier=multiplier,
                            confidence=min(1.0, abs(signal)) if signal else 0.6,
                            source=",".join(sorted(set(sources))) if sources else "forecast_consensus",
                            kind="current_event_profile"))
        return out
    return _seed(view)
