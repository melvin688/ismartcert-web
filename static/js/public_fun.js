if (typeof Array.prototype.forEach == "function") {
    //下雪方法
    (function (a) {
        function g() {
            var b, a = {};
            for (type in{
                Top: "",
                Left: ""
            }) b = "Top" == type ? "Y" : "X", "undefined" != typeof window["page" + b + "Offset"] ? a[type.toLowerCase()] = window["page" + b + "Offset"] : (b = document.documentElement.clientHeight ? document.documentElement : document.body, a[type.toLowerCase()] = b["scroll" + type]);
            return a
        }

        function o(a) {
            this.parent = document.body, this.createEl(this.parent, a), this.size = 5 * Math.random() + 5, this.el.style.width = Math.round(this.size) + "px", this.el.style.height = Math.round(this.size) + "px", this.maxLeft = document.body.offsetWidth - this.size, this.maxTop = document.body.offsetHeight - this.size, this.left = Math.random() * this.maxLeft, this.top = g().top + 1, this.angle = 1.4 + .2 * Math.random(), this.minAngle = 1.4, this.maxAngle = 1.6, this.angleDelta = .01 * Math.random(), this.speed = 2 + Math.random()
        }

        var b = window, d = function () {
            var a = document.createElement("canvas");
            return "function" == typeof a.getContext
        };
        window.Snow = function (a, b) {
            !function () {
                var c, b = ["webkit", "moz"];
                for (c = 0; c < b.length && !window.requestAnimationFrame; ++c) window.requestAnimationFrame = window[b[c] + "RequestAnimationFrame"], window.cancelAnimationFrame = window[b[c] + "CancelAnimationFrame"] || window[b[c] + "CancelRequestAnimationFrame"];
                window.requestAnimationFrame || (window.requestAnimationFrame = function (a) {
                    var c = 14, d = window.setTimeout(function () {
                        a(c)
                    }, c);
                    return d
                }), window.cancelAnimationFrame || (window.cancelAnimationFrame = function (a) {
                    clearTimeout(a)
                })
            }(), this.settings = b, this.flakes = [], this.flakeCount = b.count, this.mx = -100, this.my = -100, this.init(a)
        }, Snow.prototype.init = function (c) {
            var d, e, f, g, h, i;
            for (this.canvas = c.get(0), this.ctx = this.canvas.getContext("2d"), this.canvas.width = b.innerWidth, this.canvas.height = a(c.get(0)).attr("height"), this.flakes = [], d = 0; d < this.flakeCount; d++) e = Math.floor(Math.random() * this.canvas.width), f = Math.floor(Math.random() * this.canvas.height), g = Math.floor(100 * Math.random()) % this.settings.size + 2, h = Math.floor(100 * Math.random()) % this.settings.speed + Math.random() * g / 10 + .5, i = .5 * Math.random() + this.settings.opacity, this.flakes.push({
                speed: h,
                velY: h,
                velX: 0,
                x: e,
                y: f,
                size: g,
                stepSize: Math.random() / 30,
                step: 0,
                angle: 180,
                opacity: i
            });
            1 == this.settings.interaction && this.canvas.addEventListener("mousemove", function (a) {
                this.mx = a.clientX, this.my = a.client
            }), "string" == typeof this.settings.image && (this.image = a("<img src='" + this.settings.image + "' style='display: none'>")), this.snow()
        }, Snow.prototype.snow = function () {
            var a = this, c = function () {
                var d, e, f, g, h, i, j, k, l, m, n, o, p;
                for (a.ctx.clearRect(0, 0, a.canvas.width, a.canvas.height), d = 0; d < a.flakeCount; d++) {
                    if (e = a.flakes[d], f = a.mx, g = a.my, h = 100, i = e.x, j = e.y, k = Math.sqrt((i - f) * (i - f) + (j - g) * (j - g)), h > k) l = h / (k * k), m = (f - i) / k, n = (g - j) / k, o = l / 2, e.velX -= o * m, e.velY -= o * n; else switch (e.velX *= .98, e.velY <= e.speed && (e.velY = e.speed), a.settings.windPower) {
                        case!1:
                            e.velX += Math.cos(e.step += .05) * e.stepSize;
                            break;
                        case 0:
                            e.velX += Math.cos(e.step += .05) * e.stepSize;
                            break;
                        default:
                            e.velX += .01 + a.settings.windPower / 100
                    }
                    e.y += e.velY, e.x += e.velX, (e.y >= a.canvas.height || e.y <= 0) && a.resetFlake(e), (e.x >= a.canvas.width || e.x <= 0) && a.resetFlake(e), 0 == a.settings.image ? (p = a.ctx.createRadialGradient(e.x, e.y, 0, e.x, e.y, e.size - 1), p.addColorStop(0, a.settings.startColor), p.addColorStop(1, a.settings.endColor), a.ctx.fillStyle = p, a.ctx.beginPath(), a.ctx.arc(e.x, e.y, e.size, 0, 2 * Math.PI), a.ctx.fill()) : a.ctx.drawImage(a.image.get(0), e.x, e.y, 2 * e.size, 2 * e.size)
                }
                b.cancelAnimationFrame(c), b.requestAnimationFrame(c)
            };
            c()
        }, Snow.prototype.resetFlake = function (a) {
            var b, c, d, e;
            0 == this.settings.windPower || 0 == this.settings.windPower ? (a.x = Math.floor(Math.random() * this.canvas.width), a.y = 0) : this.settings.windPower > 0 ? (b = Array(Math.floor(Math.random() * this.canvas.width), 0), c = Array(0, Math.floor(Math.random() * this.canvas.height)), d = Array(b, c), e = d[Math.floor(Math.random() * d.length)], a.x = e[0], a.y = e[1]) : (b = Array(Math.floor(Math.random() * this.canvas.width), 0), c = Array(this.canvas.width, Math.floor(Math.random() * this.canvas.height)), d = Array(b, c), e = d[Math.floor(Math.random() * d.length)], a.x = e[0], a.y = e[1]), a.size = Math.floor(100 * Math.random()) % this.settings.size + 2, a.speed = Math.floor(100 * Math.random()) % this.settings.speed + Math.random() * a.size / 10 + .5, a.velY = a.speed, a.velX = 0, a.opacity = .5 * Math.random() + this.settings.opacity
        }, a.fn.canvasSnow = function () {
            var b = d();
            b && a(this).each(function (b, c) {
                var d = {};
                a.each(c.attributes, function (b, c) {
                    d[a.camelCase(c.name)] = Number(Number(c.value)) ? Number(c.value) : c.value
                }), "string" == typeof d.image && "false" === d.image && (d.image = !1), new Snow(a(c), {
                    speed: 1,
                    interaction: d.interaction || !0,
                    size: d.size || 2,
                    count: d.count || 200,
                    opacity: d.opacity || 1,
                    startColor: d.startColor || "rgba(255,255,255,1)",
                    endColor: d.endColor || "rgba(255,255,255,0)",
                    windPower: d.windPower || 0,
                    image: d.image || !1
                })
            })
        }, o.prototype = {
            createEl: function (a, b) {
                this.el = document.createElement("img"), this.el.classname = "nicesnowclass", this.el.setAttribute("src", b || ""), this.el.style.position = "absolute", this.el.style.display = "block", this.el.style.zIndex = "99999", this.parent.appendChild(this.el)
            }, move: function () {
                (this.angle < this.minAngle || this.angle > this.maxAngle) && (this.angleDelta = -this.angleDelta), this.angle += this.angleDelta, this.left += this.speed * Math.cos(this.angle * Math.PI), this.top -= this.speed * Math.sin(this.angle * Math.PI), this.left < 0 ? this.left = this.maxLeft : this.left > this.maxLeft && (this.left = 0)
            }, draw: function () {
                this.el.style.top = Math.round(this.top) + "px", this.el.style.left = Math.round(this.left) + "px"
            }, remove: function () {
                this.parent.removeChild(this.el), this.parent = this.el = null
            }
        }
    })(jQuery);
    //流星
    (function (a) {
        function n(e, f) {
            var g = a(e), i = document.createElement("canvas");
            g.append(i), i.width = f.width || window.innerWidth, i.height = f.height || window.innerHeight, i.className = "meteor-canvas", c = i.height, d = i.width, b = i.getContext("2d"), b.fillStyle = "black", 100 > d / h && (j = 100)
        }

        function p() {
            for (var a = 0; f > a; a++) e[a].getColor(), e[a].draw();
            clearTimeout(l), l = setTimeout(p, 200)
        }

        function r() {
          var a, d;
          for (a = 0; h > a; a++)
            (d = g[a]),
              d && (d.move(),
              d.y > c + 100 &&
                (b.clearRect(d.x, d.y - d.height, d.width, d.height),
                (i[a] = 0),
                (g[a] = new q(a)),
                g[a].init(a)));
          clearTimeout(k), (k = setTimeout(r, 10));
        }

        var b, c, d, k, l, q, e = [], f = 200, g = [], h = 3, i = [], j = 200, m = function () {
            var a = document.createElement("canvas");
            return "function" == typeof a.getContext
        }, o = function () {
            this.x = d * Math.random(), this.y = 5e3 * Math.random(), this.text = ".", this.color = "white"
        };
        o.prototype = {
            constructor: o, getColor: function () {
                var a = Math.random();
                this.color = .5 > a ? "gray" : "white"
            }, init: function () {
                this.getColor()
            }, draw: function () {
                b.beginPath(), b.arc(this.x, this.y, .05, 0, 2 * Math.PI), b.strokeStyle = this.color, b.stroke(), b.closePath()
            }
        }, q = function () {
            this.x = -1, this.y = -1, this.length = -1, this.angle = 30, this.width = -1, this.height = -1, this.speed = 1, this.offset_x = -1, this.offset_y = -1, this.alpha = 1, this.color1 = "", this.color2 = ""
        }, q.prototype = {
            constructor: q, init: function (a) {
                var b, c, d;
                this.getPos(a), this.alpha = 1, b = 80 * Math.random() + 180, this.length = Math.ceil(b), this.angle = 30, b = Math.random() + .5, this.speed = Math.ceil(b), c = Math.cos(3.14 * this.angle / 180), d = Math.sin(3.14 * this.angle / 180), this.width = this.length * c, this.height = this.length * d, this.offset_x = 3.5 * this.speed * c, this.offset_y = 3.5 * this.speed * d
            }, countPos: function () {
                this.x = this.x - this.offset_x, this.y = this.y + this.offset_y
            }, getPos: function (a) {
                function b() {
                    try {
                        if (_this.x = Math.random() * d, i.length) for (var c = 0; c < i.length; c++) if (Math.abs(_this.x - i[c]) < j) return b();
                        i[a] = _this.x
                    } catch (error) {
                        console.log(error);
                    }
                }

                _this = this, b(), this.y = .2 * c
            }, draw: function () {
                b.save(), b.beginPath(), b.lineWidth = 2.5, b.globalAlpha = this.alpha;
                var a = b.createLinearGradient(this.x, this.y, this.x + this.width, this.y - this.height);
                a.addColorStop(0, "rgba(255, 255, 255, 1)"), a.addColorStop(1, "rgba(255, 255,255 , 0)"), this.alpha < 0 && (this.alpha = -this.alpha), b.strokeStyle = a, b.moveTo(this.x, this.y), b.lineTo(this.x + this.width, this.y - this.height), b.closePath(), b.stroke(), b.restore()
            }, move: function () {
                this.x + this.width - this.offset_x;
                var c = this.y - this.height;
                this.alpha -= .002, this.countPos(), this.alpha <= 0 ? this.alpha = 0 : this.alpha > 1 && (this.alpha = 1), b.clearRect(this.x - this.offset_x, c, this.width + this.offset_x, this.height), this.draw()
            }
        }, a.fn.extend({
            canvasMeteor: function (b) {
                var c = m();
                b = b || {}, e = [], g = [], c && a(this).each(function (a, c) {
                    var d, i;
                    for (n(c, b), a = 0; f > a; a++) d = new o, d.init(), d.draw(), e.push(d);
                    for (a = 0; h > a; a++) i = new q(a), i.init(a), i.draw(), g.push(i);
                    p(), r()
                })
            }
        })
    })(jQuery);
    //烟花
    (function (a) {
        var b = function () {
            var a = document.createElement("canvas");
            return "function" == typeof a.getContext
        }, c = function (a, b) {
            return ~~(Math.random() * (b - a + 1) + a)
        }, d = function (a, b, c, d, e, f, g, h) {
            return !(e > a + c || a > e + g || f > b + d || b > f + h)
        }, e = function () {
            return window.requestAnimationFrame || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame || window.oRequestAnimationFrame || window.msRequestAnimationFrame || function (a) {
                window.setTimeout(a, 1e3 / 60)
            }
        }(), f = function (b, c) {
            this.canvas = document.createElement("canvas"), this.canvas.width = this.cw = c.width || a(window).innerWidth(), this.canvas.height = this.ch = c.height || a(window).innerHeight(), this.particles = [], this.partCount = 150, this.fireworks = [], this.mx = this.cw / 2, this.my = this.ch / 2, this.currentHue = 30, this.partSpeed = 5, this.partSpeedVariance = 10, this.partWind = 50, this.partFriction = 5, this.partGravity = 1, this.hueMin = 0, this.hueMax = 360, this.fworkSpeed = 4, this.fworkAccel = 10, this.hueVariance = 30, this.flickerDensity = 25, this.showShockwave = !0, this.showTarget = !1, this.clearAlpha = 25, a(b).append(this.canvas), this.canvas.className = "fireworks-canvas", this.ctx = this.canvas.getContext("2d"), this.ctx.lineCap = "round", this.ctx.lineJoin = "round", this.lineWidth = 1, this.bindEvents(), this.canvasLoop()
        };
        f.prototype = {
            constructor: f, createParticles: function (a, b, d) {
                for (var f, e = this.partCount; e--;) f = {
                    x: a,
                    y: b,
                    coordLast: [{x: a, y: b}, {x: a, y: b}, {x: a, y: b}],
                    angle: c(0, 360),
                    speed: c(this.partSpeed - this.partSpeedVariance <= 0 ? 1 : this.partSpeed - this.partSpeedVariance, this.partSpeed + this.partSpeedVariance),
                    friction: 1 - this.partFriction / 100,
                    gravity: this.partGravity / 2,
                    hue: c(d - this.hueVariance, d + this.hueVariance),
                    brightness: c(50, 80),
                    alpha: c(40, 100) / 100,
                    decay: c(10, 50) / 1e3,
                    wind: (c(0, this.partWind) - this.partWind / 2) / 25,
                    lineWidth: this.lineWidth
                }, this.particles.push(f)
            }, updateParticles: function () {
                for (var b, c, e, f, a = this.particles.length; a--;) b = this.particles[a], c = b.angle * Math.PI / 180, e = Math.cos(c) * b.speed, f = Math.sin(c) * b.speed, b.speed *= b.friction, b.coordLast[2].x = b.coordLast[1].x, b.coordLast[2].y = b.coordLast[1].y, b.coordLast[1].x = b.coordLast[0].x, b.coordLast[1].y = b.coordLast[0].y, b.coordLast[0].x = b.x, b.coordLast[0].y = b.y, b.x += e, b.y += f, b.y += b.gravity, b.angle += b.wind, b.alpha -= b.decay, (!d(0, 0, this.cw, this.ch, b.x - b.radius, b.y - b.radius, 2 * b.radius, 2 * b.radius) || b.alpha < .05) && this.particles.splice(a, 1)
            }, drawParticles: function () {
                for (var b, d, e, f, a = this.particles.length; a--;) b = this.particles[a], d = c(1, 3) - 1, this.ctx.beginPath(), this.ctx.moveTo(Math.round(b.coordLast[d].x), Math.round(b.coordLast[d].y)), this.ctx.lineTo(Math.round(b.x), Math.round(b.y)), this.ctx.closePath(), this.ctx.strokeStyle = "hsla(" + b.hue + ", 100%, " + b.brightness + "%, " + b.alpha + ")", this.ctx.stroke(), this.flickerDensity > 0 && (e = 50 - this.flickerDensity, c(0, e) === e && (this.ctx.beginPath(), this.ctx.arc(Math.round(b.x), Math.round(b.y), c(b.lineWidth, b.lineWidth + 3) / 2, 0, 2 * Math.PI, !1), this.ctx.closePath(), f = c(50, 100) / 100, this.ctx.fillStyle = "hsla(" + b.hue + ", 100%, " + b.brightness + "%, " + f + ")", this.ctx.fill()))
            }, createFireworks: function (a, b, d, e) {
                var f = {
                    x: a,
                    y: b,
                    startX: a,
                    startY: b,
                    hitX: !1,
                    hitY: !1,
                    coordLast: [{x: a, y: b}, {x: a, y: b}, {x: a, y: b}],
                    targetX: d,
                    targetY: e,
                    speed: this.fworkSpeed,
                    angle: Math.atan2(e - b, d - a),
                    shockwaveAngle: Math.atan2(e - b, d - a) + 90 * (Math.PI / 180),
                    acceleration: this.fworkAccel / 100,
                    hue: this.currentHue,
                    brightness: c(50, 80),
                    alpha: c(50, 100) / 100,
                    lineWidth: this.lineWidth
                };
                this.fireworks.push(f)
            }, updateFireworks: function () {
                for (var b, a = this.fireworks.length; a--;) b = this.fireworks[a], this.ctx.lineWidth = b.lineWidth, vx = Math.cos(b.angle) * b.speed, vy = Math.sin(b.angle) * b.speed, b.speed *= 1 + b.acceleration, b.coordLast[2].x = b.coordLast[1].x, b.coordLast[2].y = b.coordLast[1].y, b.coordLast[1].x = b.coordLast[0].x, b.coordLast[1].y = b.coordLast[0].y, b.coordLast[0].x = b.x, b.coordLast[0].y = b.y, b.startX >= b.targetX ? b.x + vx <= b.targetX ? (b.x = b.targetX, b.hitX = !0) : b.x += vx : b.x + vx >= b.targetX ? (b.x = b.targetX, b.hitX = !0) : b.x += vx, b.startY >= b.targetY ? b.y + vy <= b.targetY ? (b.y = b.targetY, b.hitY = !0) : b.y += vy : b.y + vy >= b.targetY ? (b.y = b.targetY, b.hitY = !0) : b.y += vy, b.hitX && b.hitY && (this.createParticles(b.targetX, b.targetY, b.hue), this.fireworks.splice(a, 1))
            }, drawFireworks: function () {
                var b, d, a = this.fireworks.length;
                for (this.ctx.globalCompositeOperation = "lighter"; a--;) b = this.fireworks[a], this.ctx.lineWidth = b.lineWidth, d = c(1, 3) - 1, this.ctx.beginPath(), this.ctx.moveTo(Math.round(b.coordLast[d].x), Math.round(b.coordLast[d].y)), this.ctx.lineTo(Math.round(b.x), Math.round(b.y)), this.ctx.closePath(), this.ctx.strokeStyle = "hsla(" + b.hue + ", 100%, " + b.brightness + "%, " + b.alpha + ")", this.ctx.stroke(), this.showTarget && (this.ctx.save(), this.ctx.beginPath(), this.ctx.arc(Math.round(b.targetX), Math.round(b.targetY), c(1, 8), 0, 2 * Math.PI, !1), this.ctx.closePath(), this.ctx.lineWidth = 1, this.ctx.stroke(), this.ctx.restore()), this.showShockwave && (this.ctx.save(), this.ctx.translate(Math.round(b.x), Math.round(b.y)), this.ctx.rotate(b.shockwaveAngle), this.ctx.beginPath(), this.ctx.arc(0, 0, 1 * (b.speed / 5), 0, Math.PI, !0), this.ctx.strokeStyle = "hsla(" + b.hue + ", 100%, " + b.brightness + "%, " + c(25, 60) / 100 + ")", this.ctx.lineWidth = b.lineWidth, this.ctx.stroke(), this.ctx.restore())
            }, bindEvents: function () {
                function d() {
                    a.mx = a.canvas.width * Math.random() - a.canvas.offsetLeft, a.my = a.canvas.height * Math.random() - a.canvas.offsetTop, a.currentHue = c(a.hueMin, a.hueMax), a.createFireworks(a.cw / 2, a.ch, a.mx, a.my), setTimeout(function () {
                        e(d.bind(a))
                    }, 500)
                }

                var a = this;
                a.clear(), e(d.bind(a))
            }, clear: function () {
                this.particles = [], this.fireworks = [], this.ctx.clearRect(0, 0, this.cw, this.ch)
            }, canvasLoop: function () {
                var a = this;
                e(this.canvasLoop.bind(a), this.canvas), this.ctx.globalCompositeOperation = "destination-out", this.ctx.fillStyle = "rgba(0, 0, 0, " + this.clearAlpha / 100 + ")", this.ctx.fillRect(0, 0, this.cw, this.ch), this.updateFireworks(), this.updateParticles(), this.drawFireworks(), this.drawParticles()
            }
        }, a.fn.extend({
            canvasFireworks: function (c) {
                var d = b();
                c = c || {}, d && a(this).each(function (a, b) {
                    new f(b, c)
                })
            }
        })
    })(jQuery);
    //下雨
    (function (a) {
        function b() {
            this.dir = 3, this.height = 30, this.count = 100
        }

        b.prototype.init = function () {
            var a = this;
            a.canvas = document.createElement("canvas"), a.canvas.width = a.W = a.curDom.width(), a.canvas.height = a.H = a.curDom.height(), a.curDom.append(a.canvas), a.ctx = a.canvas.getContext("2d"), setInterval(function () {
                a.clearCanvas()
            }, 100)
        }, b.prototype.clearCanvas = function () {
            var a = this;
            a.ctx.clearRect(0, 0, a.W, a.H), a.draws()
        }, b.prototype.draw = function (a, b) {
            var c = this, d = c.ctx, e = c.ctx.createLinearGradient(a, b, a - c.dir, b + c.height);
            e.addColorStop(0, "rgba(0,0,0,0)"), e.addColorStop(.5, "rgba(105,105,105,0.2)"), e.addColorStop(1, "rgba(255,255,255,0.8)"), d.strokeStyle = e, d.beginPath(), d.moveTo(a, b), d.lineTo(a - c.dir, b + c.height), d.lineWidth = 2, d.stroke(), d.closePath()
        }, b.prototype.draws = function () {
            var b, a = this;
            for (b = 1; b <= a.count; b++) a.draw(Math.random() * a.W, Math.random() * a.H)
        }, a.fn.extend({
            canvasRain: function (c) {
                var d = new b;
                d.curDom = a(this), c && (c.dir && (d.dir = c.dir), c.height && (d.height = c.height), c.count && (d.count = c.count)), d.init()
            }
        })
    })(jQuery);
    //气泡
    (function ($) {
        $.fn.canvasAirBubble = function (options) {
            var curDom = $(this), width = curDom.width(), height = curDom.height(), canvas, ctx, target,
                animateHeader = true, circles = [], settings = $.extend({
                    color: 'rgba(255,255,255,.4)',
                    radius: 20,
                    densety: 0.3,
                    clearOffset: 0.2
                }, options);
            initContainer();

            function initContainer() {
                target = {x: 0, y: height};
                initCanvas();
                canvas = document.getElementById('canvas');
                canvas.width = width;
                canvas.height = height;
                ctx = canvas.getContext('2d');
                for (var x = 0; x < width * settings.densety; x++) {
                    var c = new Circle();
                    circles.push(c)
                }
                animate()
            }

            function initCanvas() {
                var canvasElement = document.createElement('canvas');
                canvasElement.id = 'canvas';
                curDom.append(canvasElement)
            }

            function animate() {
                if (animateHeader) {
                    ctx.clearRect(0, 0, width, height);
                    for (var i in circles) {
                        circles[i].draw()
                    }
                }
                requestAnimationFrame(animate)
            }

            function randomColor() {
                return "rgba(" + Math.floor(Math.random() * 255) + "," + Math.floor(Math.random() * 255) + "," + Math.floor(Math.random() * 255) + "," + Math.random().toPrecision(2) + ")"
            }

            function Circle() {
                var self = this;
                (function () {
                    self.pos = {};
                    init()
                })();

                function init() {
                    self.pos.x = Math.random() * width;
                    self.pos.y = height + Math.random() * 100;
                    self.alpha = 0.1 + Math.random() * settings.clearOffset;
                    self.scale = 0.1 + Math.random() * 0.3;
                    self.speed = Math.random();
                    if (settings.color == 'random') {
                        self.color = randomColor()
                    } else {
                        self.color = settings.color
                    }
                }

                this.draw = function () {
                    if (self.alpha <= 0) {
                        init()
                    }
                    self.pos.y -= self.speed;
                    self.alpha -= 0.0005;
                    ctx.beginPath();
                    ctx.arc(self.pos.x, self.pos.y, self.scale * settings.radius, 0, 2 * Math.PI, false);
                    ctx.fillStyle = self.color;
                    ctx.fill();
                    ctx.closePath()
                }
            }
        }
    })(jQuery);
}
/**************** 前台页面公用的js方法 *******************/
/*Banner3D的插件*/
;(function($) {
    function Banner3D(curDom) {
        var that = this;
        that.curDom = curDom;
        that.curDomType = that.curDom.hasClass('customModuleRowGroup') ? 'rowGroup' : 'banner';
        that.effectType = that.curDom.attr('data-effect-type');
        that.autoTime = (that.curDomType === 'rowGroup' && is_action) ? null : that.curDom.attr('data-auto-time') * 1000;
        that.autoTimeFun = null;
        that.durationTime = that.curDom.attr('data-duration-time') * 1000;
        that.canvasType = that.curDom.attr('data-canvas-type');
        that.piclistArea = that.curDomType === 'rowGroup' ? that.curDom.find(".rowGroup-box") : that.curDom.find(".piclist-area");
        that.piclistBox = that.curDomType === 'rowGroup' ? that.curDom.find(".rowGroup-box-inner") : that.curDom.find(".piclist-box");
        that.btnArea = that.curDom.children(".ctrl-btn-area");
        that.sidebarArea = that.curDom.children(".ctrl-sidebar-area");
        that.smallpicObj = {
            'area': that.curDom.children(".smallpic-area")
        };
        that.liArray = that.curDomType === 'rowGroup' ? that.curDom.find(".customModuleRow") : that.piclistBox.find(".list-item");
        that.sumLi = that.liArray.length;
        that.curLiIndex = null;
        that.curLi = null;
        that.defaultLi = that.curDom.attr('data-showsum') || 0;
        that.effectKey = 0;
        that.targetIndex = 0;
    }
    Banner3D.prototype.init = function() {
        var that = this,
            curClass = 'cur';
        that.liArray.css({
            'animation-duration': that.durationTime + 'ms'
        });
        that.liArray.eq(that.defaultLi).addClass(curClass);
        that.liArray.each(function(i, dom) {
            dom = $(dom);
            dom.data('loaded', 0).attr('data-loaded', 0);
            if (dom.hasClass(curClass)) {
                that.curLiIndex = that.targetIndex = i;
                that.curLi = dom;
                that.sidebarArea.length && that.sidebarArea.find('em').eq(i).addClass('cur');
                that.smallpicObj.area.length && that.smallpicObj.area.find('em').eq(i).addClass('cur');
                that.loadImg();
            }
        });
        that.btnArea.length && that.btnArea.on({
            click: function() {
                var atcion = $(this).attr('data-action');
                that.effectKey = atcion == 'next' ? 1 : 0;
                that.initFun(that.curLiIndex + (atcion == 'next' ? +1 : -1));
            }
        }, '.btn');
        that.sidebarArea.length && that.sidebarArea.on({
            click: function() {
                var $this = $(this);
                that.autoTime = null;
                that.autoTimeFun && clearTimeout(that.autoTimeFun);
                if (!$this.hasClass('cur')) {
                    var index = $this.index();
                    that.effectKey = index > that.curLiIndex ? 1 : 0;
                    that.initFun(index);
                }
            }
        }, 'em');
        that.curDom.on({
            'mouseenter': function() {
                that.autoTime = null;
                clearTimeout(that.autoTimeFun);
            },
            'mouseleave': function() {
                that.autoTime = (that.curDomType === 'rowGroup' && is_action) ? null : that.curDom.attr('data-auto-time') * 1000;
                that.autoPlay();
            }
        });
        // 焦点图特出处理
        if (that.curDomType === 'banner') {
            if (that.smallpicObj.area.length) {
                that.smallpicObj.area.on({
                    click: function() {
                        var $this = $(this);
                        that.autoTime = null;
                        that.autoTimeFun && clearTimeout(that.autoTimeFun);
                        if (!$this.hasClass('cur')) {
                            var index = $this.index();
                            that.effectKey = index > that.curLiIndex ? 1 : 0;
                            that.initFun(index);
                        }
                    }
                }, 'em');
                that.smallpicAnimation();
            }
            setTimeout(function(){
                wsf.banner3D.canvasLoad(that.curDom);
            }, 100);
        }
        that.autoPlay();
    };
    Banner3D.prototype.loadImg = function() {
        var that = this;
        if(that.curLi.length){
            if(that.curLi.attr('data-loaded') * 1 !== 1){
                that.curLi.attr('data-loaded', 1);
                if (that.curDomType === 'rowGroup') {
                    that.curLi.find('.ev-module-edit,.customModule').each(function(i, dom) {
                        wsf.f.upJsFun($(dom));
                    });
                } else {
                    var img = new Image(),
                        picItem = that.curLi.find('.pic-item');
                    picItem.addClass('loading-pic');
                    if (picItem.data('title')) {
                        picItem.attr('title', picItem.data('title'));
                    }
                    img.onload = function() {
                        picItem.css('background-image', 'url(' + picItem.data('imgurl') + ')');
                        img = null;
                        picItem.removeClass('loading-pic');
                    };
                    img.src = picItem.data('imgurl');
                }
            }
            if (that.curDomType === 'rowGroup') {
                setTimeout(function(){
                    that.curLi.find('.ev-module-edit,.customModule').each(function(i, dom) {
                        dom = $(dom);
                        var curDomEditBox = dom;
                        if (dom.hasClass('ev-module-edit')) {
                            curDomEditBox = dom.find('div.ev-module-edit-box:first')
                        }
                        if (curDomEditBox.data('animate-name')) {
                            wsf.f.animationLoadMo(curDomEditBox);
                        }
                        wsf.f.imgLazyLoading(dom);
                    });
                    that.liArray.each(function(i, rowDom) {
                        rowDom = $(rowDom);
                        if (!rowDom.is(that.curLi)) {
                            rowDom.find('.ev-module-edit,.customModule').each(function(i, dom) {
                                dom = $(dom);
                                var curDomEditBox = dom;
                                if (dom.hasClass('ev-module-edit')) {
                                    curDomEditBox = dom.find('div.ev-module-edit-box:first');
                                }
                                if (curDomEditBox.data('animate-name')) {
                                    curDomEditBox.attr('data-loadanimate', 0);
                                    curDomEditBox.addClass('load-animate');
                                }
                            });
                        }
                    });
                },that.durationTime);
            }
        }
    };
    Banner3D.prototype.smallpicAnimation = function() {
        var that = this;
        if (!that.smallpicObj.inner) {
            that.smallpicObj.areaW = that.smallpicObj.area.width();
            that.smallpicObj.areaH = that.smallpicObj.area.height();
            that.smallpicObj.inner = that.smallpicObj.area.find('.smallpic-inner');
            that.smallpicObj.inner.data('translateX', 0);
            that.smallpicObj.emArray = that.smallpicObj.inner.find('em');
            if ($.inArray(that.smallpicObj.area.attr('data-skin') * 1, [151, 152]) == -1) {
                that.smallpicObj.innerW = (function() {
                    var w = that.smallpicObj.areaW,
                        w_ = 0;
                    that.smallpicObj.emArray.each(function(i, dom) {
                        dom = $(dom);
                        w_ += dom.width();
                    });
                    return Math.max(w, w_);
                })();
                that.smallpicObj.inner.width(that.smallpicObj.innerW);
            }
        }
        if ($.inArray(that.smallpicObj.area.attr('data-skin') * 1, [151, 152]) == -1) {
            if (that.smallpicObj.areaW < that.smallpicObj.innerW) {
                (function() {
                    var cz = that.smallpicObj.innerW - that.smallpicObj.areaW,
                        curEm = that.smallpicObj.emArray.eq(that.targetIndex),
                        left = curEm.position().left,
                        x = 0;
                    if (left < 0 || left + curEm.width() > that.smallpicObj.areaW) {
                        x = that.smallpicObj.inner.data('translateX') + (left < 0 ? Math.abs(left) : -(curEm.width() + left - that.smallpicObj.areaW));
                        left < 0 ? (x > 0 && (x = 0)) : ((Math.abs(x) > that.smallpicObj.innerW - that.smallpicObj.areaW) && (x = -(that.smallpicObj.innerW - that.smallpicObj.areaW)));
                        that.smallpicObj.inner.data('translateX', x);
                        that.smallpicObj.inner.css({
                            'transform': 'translateX(' + x + 'px)'
                        });
                    }
                })();
            }
        }
    };
    Banner3D.prototype.initFun = function(targetIndex) {
        var that = this;
        if(is_action && that.curDomType === 'rowGroup'){
          that.liArray = that.curDom.find(".customModuleRow");
          that.sumLi = that.liArray.length;
          that.effectType = that.curDom.attr('data-effect-type');
          that.targetIndex = that.curLiIndex = that.curDom.find(".customModuleRow.cur").index();
        }
        // that.autoTime != null && that.curDom.attr('data-auto-time') * 1000;
        if (that.durationTime != that.curDom.attr('data-duration-time') * 1000) {
            that.durationTime = that.curDom.attr('data-duration-time') * 1000;
            that.liArray.css({
                'animation-duration': that.durationTime + 'ms'
            });
        }
        if (that.targetIndex == that.curLiIndex && that.sumLi > 1) {
            that.targetIndex = ((targetIndex == that.sumLi && '0') || (targetIndex < 0 && (that.sumLi - 1)) || targetIndex) * 1;
            that.curLi = $(that.liArray[that.targetIndex]);
            if (typeof Array.prototype.forEach != "function" && $.inArray($.browser.version, ['6.0', '7.0', '8.0', '9.0', '10.0']) != -1) {
                that.IEanimationFun((that.effectType == 0 && Math.floor(Math.random() * 3 + 1)) || that.effectType * 1);
            } else {
                that.animationFun((that.effectType == 0 && Math.floor(Math.random() * 8 + 1)) || that.effectType * 1);
            }
          that.loadImg();
        }
    };
    Banner3D.prototype.effectJson = {
        1: [
            ['bannerEffect-1-1-out', 'bannerEffect-1-1-in'],
            ['bannerEffect-1-2-out', 'bannerEffect-1-2-in']
        ],
        2: [
            ['bannerEffect-2-2-out', 'bannerEffect-2-2-in'],
            ['bannerEffect-2-1-out', 'bannerEffect-2-1-in']
        ],
        3: [
            ['bannerEffect-3-out', 'bannerEffect-3-in'],
            ['bannerEffect-3-out', 'bannerEffect-3-in']
        ],
        4: [
            ['bannerEffect-4-1-out', 'bannerEffect-4-1-in'],
            ['bannerEffect-4-2-out', 'bannerEffect-4-2-in']
        ],
        5: [
            ['bannerEffect-5-1-out', 'bannerEffect-5-1-in'],
            ['bannerEffect-5-2-out', 'bannerEffect-5-2-in']
        ],
        6: [
            ['bannerEffect-6-1-out', 'bannerEffect-6-1-in'],
            ['bannerEffect-6-2-out', 'bannerEffect-6-2-in']
        ],
        7: [
            ['bannerEffect-7-1-out', 'bannerEffect-7-1-in'],
            ['bannerEffect-7-2-out', 'bannerEffect-7-2-in']
        ],
        8: [
            ['bannerEffect-8-1-out', 'bannerEffect-8-1-in'],
            ['bannerEffect-8-2-out', 'bannerEffect-8-2-in']
        ]
    };
    Banner3D.prototype.animationFun = function(types) {
        var that = this,
            effectKey = that.effectKey,
            liArray = that.liArray,
            effectJson = that.effectJson;
        that.sidebarArea.length && that.sidebarArea.find('em').removeClass('cur').eq(that.targetIndex).addClass('cur');
        that.smallpicObj.area.length && (that.smallpicObj.area.find('em').removeClass('cur').eq(that.targetIndex).addClass('cur'), that.smallpicAnimation());
        that.piclistArea.attr('class', (that.curDomType === 'rowGroup' ? 'rowGroup-box' : 'piclist-area') + ' bannerEffect-' + types);
        switch (types) {
            case 6:
                liArray.css({
                    '-webkit-transform-origin': '50% 50% -' + (that.piclistArea.width() / 2) + 'px',
                    '-moz-transform-origin': '50% 50% -' + (that.piclistArea.width() / 2) + 'px',
                    '-ms-transform-origin': '50% 50% -' + (that.piclistArea.width() / 2) + 'px',
                    '-o-transform-origin': '50% 50% -' + (that.piclistArea.width() / 2) + 'px',
                    'transform-origin': '50% 50% -' + (that.piclistArea.width() / 2) + 'px'
                });
                break;
            case 7:
                liArray.css({
                    '-webkit-transform-origin': '50% 50% -' + (that.piclistArea.height() / 2) + 'px',
                    '-moz-transform-origin': '50% 50% -' + (that.piclistArea.height() / 2) + 'px',
                    '-ms-transform-origin': '50% 50% -' + (that.piclistArea.height() / 2) + 'px',
                    '-o-transform-origin': '50% 50% -' + (that.piclistArea.height() / 2) + 'px',
                    'transform-origin': '50% 50% -' + (that.piclistArea.height() / 2) + 'px'
                });
                break;
            default:
                liArray.css({
                    '-webkit-transform-origin': '50% 50%',
                    '-moz-transform-origin': '50% 50%',
                    '-ms-transform-origin': '50% 50%',
                    '-o-transform-origin': '50% 50%',
                    'transform-origin': '50% 50%'
                });
                break;
        }
        switch (types) {
            case 1:
            case 2:
            case 3:
            case 6:
            case 7:
            case 8:
                liArray.eq(that.curLiIndex).addClass(effectJson[types][effectKey][0]);
                liArray.eq(that.targetIndex).addClass('cur ' + effectJson[types][effectKey][1]);
                setTimeout(function() {
                    that.curDomType !== 'rowGroup' && (function() {
                        var curVideo = liArray.eq(that.curLiIndex).find('video'),
                            nextVideo = liArray.eq(that.targetIndex).find('video');
                        curVideo.length && curVideo.attr('src') != 'none' && curVideo[0].pause();
                        nextVideo.length && nextVideo.attr('src') != 'none' && nextVideo[0].play();

                    })();
                    liArray.eq(that.curLiIndex).removeClass('cur ' + effectJson[types][effectKey][0]);
                    liArray.eq(that.targetIndex).removeClass(effectJson[types][effectKey][1]);
                    that.curLiIndex = that.targetIndex;
                    that.autoPlay();
                }, that.durationTime - 20);
                break;
            case 4:
            case 5:
                liArray.eq(that.curLiIndex).addClass(effectJson[types][effectKey][0]);
                setTimeout(function() {
                    liArray.eq(that.curLiIndex).removeClass('cur ' + effectJson[types][effectKey][0]);
                    liArray.eq(that.targetIndex).addClass('cur ' + effectJson[types][effectKey][1]);
                    setTimeout(function() {
                        (function() {
                            var curVideo = liArray.eq(that.curLiIndex).find('video'),
                                nextVideo = liArray.eq(that.targetIndex).find('video');
                            curVideo.length && curVideo.attr('src') != 'none' && curVideo[0].pause();
                            nextVideo.length && nextVideo.attr('src') != 'none' && nextVideo[0].play();

                        })();
                        liArray.eq(that.targetIndex).removeClass(effectJson[types][effectKey][1]);
                        that.curLiIndex = that.targetIndex;
                        that.autoPlay();
                    }, that.durationTime);
                }, that.durationTime - 20);
                break;
        }
    };
    Banner3D.prototype.IEanimationFun = function(types) {
        var that = this,
            aKey = null,
            effectKey = that.effectKey,
            liArray = that.liArray,
            effectJson = that.effectJson;
        that.sidebarArea.length && that.sidebarArea.find('em').removeClass('cur').eq(that.targetIndex).addClass('cur');
        that.smallpicObj.area.length && that.smallpicObj.area.find('em').removeClass('cur').eq(that.targetIndex).addClass('cur');
        that.piclistArea.attr('class', (that.curDomType === 'rowGroup' ? 'rowGroup-box' : 'piclist-area') + ' bannerEffect-' + types);
        switch (types) {
            case 1:
                liArray.eq(that.curLiIndex).animate({
                    'opacity': 0
                }, that.durationTime, function() {
                    $(this).removeClass('cur').css('opacity', '');
                    that.curLiIndex = that.targetIndex;
                    that.autoPlay();
                });
                liArray.eq(that.targetIndex).addClass('cur').css('opacity', 0).animate({
                    'opacity': 1
                }, that.durationTime);
                break;
            case 2:
                aKey = that.effectKey == 1 ? ['-100%', '100%'] : ['100%', '-100%'];
                liArray.eq(that.curLiIndex).animate({
                    'left': aKey[0]
                }, that.durationTime, function() {
                    $(this).removeClass('cur').css('left', 0);
                    that.curLiIndex = that.targetIndex;
                    that.autoPlay();
                });
                liArray.eq(that.targetIndex).addClass('cur').css('left', aKey[1]).animate({
                    'left': 0
                }, that.durationTime);
                break;
            case 3:
                aKey = that.effectKey == 1 ? ['-100%', '100%'] : ['100%', '-100%'];
                liArray.eq(that.curLiIndex).animate({
                    'top': aKey[0]
                }, that.durationTime, function() {
                    $(this).removeClass('cur').css('top', 0);
                    that.curLiIndex = that.targetIndex;
                    that.autoPlay();
                });
                liArray.eq(that.targetIndex).addClass('cur').css('top', aKey[1]).animate({
                    'top': 0
                }, that.durationTime);
                break;
        }
    };
    Banner3D.prototype.autoPlay = function() {
        var that = this;
        if (that.autoTime) {
            that.autoTimeFun = setTimeout(function() {
                that.effectKey = 1;
                that.initFun(that.curLiIndex + 1);
                // that.curLi = that.liArray.eq(that.curLiIndex + 1);
            }, that.autoTime);
        }
    };
    $.extend({
        banner3D: function(curDom) {
            curDom[0]['banner3D'] = new Banner3D(curDom);
            curDom[0].banner3D.init();
        }
    });
})(jQuery);
/*
 *   jquery focusImg
 *   焦点图js功能
 */
(function ($) {
    $.fn.extend({
        focusImg: function (options) {
            return this.each(function () {
                var Opts = {
                    uistyle: "style-1",
                    fnclass: "inOut",
                    evtype: "mouseenter",
                    usertime: 6
                };
                Opts = $.extend(Opts, options);

                var full = $(this),
                    fWidth = full.parent().width(),
                    fHeight = full.parent().height(),
                    uistyle = Opts.uistyle,
                    fnclass = Opts.fnclass,
                    pUl = full.children("ul"),
                    pLi = pUl.find("li"),
                    bNav = $("<div></div>"),
                    times = null,
                    an = true,
                    y = 0,
                    change = function (e) { //图片地址更换
                        var curli = e,
                            bigpic = curli.attr("bigpic"),
                            bcolor = curli.attr("bcolor");
                        curli.css({
                            "background-color": bcolor,
                            "background-image": "url(" + bigpic + ")"
                        });
                        curli.attr("change", "true");
                    },
                    eventfn = function (obj, fobj) { //事件模式
                        obj.find(fobj).each(function (e) {
                            $(this).bind(Opts.evtype, function () {
                                eval(fnclass + "(e)");
                            });
                        });
                    };
                switch (Opts.fnclass) {
                    case 'inOut':
                        (function () {
                            pUl.addClass('banner-pic-1').css({
                                "height": fHeight
                            });
                            pLi.each(function () {
                                $(this).css({
                                    "height": fHeight
                                });
                            });
                        })();
                        break;
                    case 'LMove':
                        (function () {
                            full.css({
                                "height": fHeight + "px"
                            });
                            pUl.addClass('banner-pic-2').css({
                                "width": pLi.length * fWidth + "px",
                                "height": fHeight + "px"
                            });
                            pLi.each(function () {
                                $(this).css({
                                    "width": fWidth + "px",
                                    "height": fHeight
                                });
                            });
                        })();
                        break;
                    case 'TMove':
                        (function () {
                            full.css({
                                "height": fHeight + "px"
                            });
                            pUl.addClass('banner-pic-3').css({
                                "height": pLi.length * fHeight + "px"
                            });
                            pLi.each(function () {
                                $(this).css({
                                    "height": fHeight
                                });
                            });
                        })();
                        break;
                }
                switch (Opts.uistyle) {
                    case "style-1":
                        if (pLi.length > 1) {
                            var ban = $("<div class='b-nav'></div>");
                            pLi.each(function (e) {
                                ban.append("<span></span>");
                            });
                            full.append(bNav.attr("class", "banner-nav-1").html(ban));
                            eventfn(ban, "span");
                        }
                        break;
                    case "style-2":
                        if (pLi.length > 1) {
                            var ban = $("<div class='b-nav'></div>");
                            pLi.each(function (e) {
                                ban.append("<span>" + (e + 1) + "</span>");
                            });
                            full.append(bNav.attr("class", "banner-nav-2").html(ban));
                            eventfn(ban, "span");
                        }
                        break;
                    case "style-3":
                        var ban = $("<div class='b-nav'></div>");
                        pLi.each(function () {
                            var smallurl = $(this).attr("smallpic") ? $(this).attr("smallpic") : $(this).attr("bigpic");
                            var s = $("<span></span>").html("<img src=" + smallurl + " />");
                            ban.append(s);
                        });
                        full.append(bNav.attr("class", "banner-nav-3").html(ban));
                        eventfn(ban, "span");
                        break;
                }
                var inOut = function (e) {
                    var curli = pLi.eq(e);
                    if (curli.attr("change") == "false") {
                        change(curli);
                    }
                    var ospan = bNav.find("span.cur");
                    var o = ospan.index();
                    if (e != o) {
                        if (an) {
                            an = false;
                            ospan.removeClass("cur");
                            bNav.find("span:eq(" + e + ")").addClass("cur");
                            pLi.eq(o).css({
                                "z-index": 3
                            }).animate({
                                "opacity": 0
                            }, 1000, function () {
                                an = true;
                                $(this).css({
                                    "opacity": 1,
                                    "z-index": 1
                                });
                            });
                            curli.css({
                                "display": "block",
                                "z-index": 2
                            });
                            y = y + 1;
                            if (y >= pLi.length) {
                                y = 0;
                            }
                        }
                    }
                };
                var LMove = function (e) {
                    var fullW = full.parent().width(),
                        fullH = full.parent().height();
                    var curli = pLi.eq(e);
                    if (curli.attr("change") == "false") {
                        change(curli);
                    }
                    var ospan = bNav.find("span.cur");
                    var o = ospan.index();
                    if (an) {
                        an = false;
                        ospan.removeClass("cur");
                        bNav.find("span:eq(" + e + ")").addClass("cur");
                        pUl.animate({
                            "left": "-" + e * fullW
                        }, 200, function () {
                            an = true;
                            y = y + 1;
                            if (y >= pLi.length) {
                                y = 0;
                            }
                        });
                    }
                };
                var TMove = function (e) {
                    var fullH = full.height();
                    var curli = pLi.eq(e);
                    if (curli.attr("change") == "false") {
                        change(curli);
                    }
                    var ospan = bNav.find("span.cur");
                    var o = ospan.index();
                    if (an) {
                        an = false;
                        ospan.removeClass("cur");
                        bNav.find("span:eq(" + e + ")").addClass("cur");
                        pUl.animate({
                            "top": "-" + e * fullH
                        }, 200, function () {
                            an = true;
                            y = y + 1;
                            if (y >= pLi.length) {
                                y = 0;
                            }
                        });
                    }
                };
                times = setInterval(function () {
                    eval(Opts.fnclass + "(y)");
                }, (Opts.usertime) * 1000);
                full.mouseenter(function () {
                    clearInterval(times);
                });
                full.mouseleave(function () {
                    times = setInterval(function () {
                        eval(Opts.fnclass + "(y)");
                    }, (Opts.usertime) * 1000);
                });
                eval(Opts.fnclass + "(y)");
            });
        }
    });
})(jQuery);
/*生成二维码得方法*/
(function(r){r.fn.qrcode=function(h){var s;function u(a){this.mode=s;this.data=a}function o(a,c){this.typeNumber=a;this.errorCorrectLevel=c;this.modules=null;this.moduleCount=0;this.dataCache=null;this.dataList=[]}function q(a,c){if(void 0==a.length)throw Error(a.length+"/"+c);for(var d=0;d<a.length&&0==a[d];)d++;this.num=Array(a.length-d+c);for(var b=0;b<a.length-d;b++)this.num[b]=a[b+d]}function p(a,c){this.totalCount=a;this.dataCount=c}function t(){this.buffer=[];this.length=0}u.prototype={getLength:function(){return this.data.length},
    write:function(a){for(var c=0;c<this.data.length;c++)a.put(this.data.charCodeAt(c),8)}};o.prototype={addData:function(a){this.dataList.push(new u(a));this.dataCache=null},isDark:function(a,c){if(0>a||this.moduleCount<=a||0>c||this.moduleCount<=c)throw Error(a+","+c);return this.modules[a][c]},getModuleCount:function(){return this.moduleCount},make:function(){if(1>this.typeNumber){for(var a=1,a=1;40>a;a++){for(var c=p.getRSBlocks(a,this.errorCorrectLevel),d=new t,b=0,e=0;e<c.length;e++)b+=c[e].dataCount;
    for(e=0;e<this.dataList.length;e++)c=this.dataList[e],d.put(c.mode,4),d.put(c.getLength(),j.getLengthInBits(c.mode,a)),c.write(d);if(d.getLengthInBits()<=8*b)break}this.typeNumber=a}this.makeImpl(!1,this.getBestMaskPattern())},makeImpl:function(a,c){this.moduleCount=4*this.typeNumber+17;this.modules=Array(this.moduleCount);for(var d=0;d<this.moduleCount;d++){this.modules[d]=Array(this.moduleCount);for(var b=0;b<this.moduleCount;b++)this.modules[d][b]=null}this.setupPositionProbePattern(0,0);this.setupPositionProbePattern(this.moduleCount-
    7,0);this.setupPositionProbePattern(0,this.moduleCount-7);this.setupPositionAdjustPattern();this.setupTimingPattern();this.setupTypeInfo(a,c);7<=this.typeNumber&&this.setupTypeNumber(a);null==this.dataCache&&(this.dataCache=o.createData(this.typeNumber,this.errorCorrectLevel,this.dataList));this.mapData(this.dataCache,c)},setupPositionProbePattern:function(a,c){for(var d=-1;7>=d;d++)if(!(-1>=a+d||this.moduleCount<=a+d))for(var b=-1;7>=b;b++)-1>=c+b||this.moduleCount<=c+b||(this.modules[a+d][c+b]=
    0<=d&&6>=d&&(0==b||6==b)||0<=b&&6>=b&&(0==d||6==d)||2<=d&&4>=d&&2<=b&&4>=b?!0:!1)},getBestMaskPattern:function(){for(var a=0,c=0,d=0;8>d;d++){this.makeImpl(!0,d);var b=j.getLostPoint(this);if(0==d||a>b)a=b,c=d}return c},createMovieClip:function(a,c,d){a=a.createEmptyMovieClip(c,d);this.make();for(c=0;c<this.modules.length;c++)for(var d=1*c,b=0;b<this.modules[c].length;b++){var e=1*b;this.modules[c][b]&&(a.beginFill(0,100),a.moveTo(e,d),a.lineTo(e+1,d),a.lineTo(e+1,d+1),a.lineTo(e,d+1),a.endFill())}return a},
    setupTimingPattern:function(){for(var a=8;a<this.moduleCount-8;a++)null==this.modules[a][6]&&(this.modules[a][6]=0==a%2);for(a=8;a<this.moduleCount-8;a++)null==this.modules[6][a]&&(this.modules[6][a]=0==a%2)},setupPositionAdjustPattern:function(){for(var a=j.getPatternPosition(this.typeNumber),c=0;c<a.length;c++)for(var d=0;d<a.length;d++){var b=a[c],e=a[d];if(null==this.modules[b][e])for(var f=-2;2>=f;f++)for(var i=-2;2>=i;i++)this.modules[b+f][e+i]=-2==f||2==f||-2==i||2==i||0==f&&0==i?!0:!1}},setupTypeNumber:function(a){for(var c=
        j.getBCHTypeNumber(this.typeNumber),d=0;18>d;d++){var b=!a&&1==(c>>d&1);this.modules[Math.floor(d/3)][d%3+this.moduleCount-8-3]=b}for(d=0;18>d;d++)b=!a&&1==(c>>d&1),this.modules[d%3+this.moduleCount-8-3][Math.floor(d/3)]=b},setupTypeInfo:function(a,c){for(var d=j.getBCHTypeInfo(this.errorCorrectLevel<<3|c),b=0;15>b;b++){var e=!a&&1==(d>>b&1);6>b?this.modules[b][8]=e:8>b?this.modules[b+1][8]=e:this.modules[this.moduleCount-15+b][8]=e}for(b=0;15>b;b++)e=!a&&1==(d>>b&1),8>b?this.modules[8][this.moduleCount-
    b-1]=e:9>b?this.modules[8][15-b-1+1]=e:this.modules[8][15-b-1]=e;this.modules[this.moduleCount-8][8]=!a},mapData:function(a,c){for(var d=-1,b=this.moduleCount-1,e=7,f=0,i=this.moduleCount-1;0<i;i-=2)for(6==i&&i--;;){for(var g=0;2>g;g++)if(null==this.modules[b][i-g]){var n=!1;f<a.length&&(n=1==(a[f]>>>e&1));j.getMask(c,b,i-g)&&(n=!n);this.modules[b][i-g]=n;e--; -1==e&&(f++,e=7)}b+=d;if(0>b||this.moduleCount<=b){b-=d;d=-d;break}}}};o.PAD0=236;o.PAD1=17;o.createData=function(a,c,d){for(var c=p.getRSBlocks(a,
    c),b=new t,e=0;e<d.length;e++){var f=d[e];b.put(f.mode,4);b.put(f.getLength(),j.getLengthInBits(f.mode,a));f.write(b)}for(e=a=0;e<c.length;e++)a+=c[e].dataCount;if(b.getLengthInBits()>8*a)throw Error("code length overflow. ("+b.getLengthInBits()+">"+8*a+")");for(b.getLengthInBits()+4<=8*a&&b.put(0,4);0!=b.getLengthInBits()%8;)b.putBit(!1);for(;!(b.getLengthInBits()>=8*a);){b.put(o.PAD0,8);if(b.getLengthInBits()>=8*a)break;b.put(o.PAD1,8)}return o.createBytes(b,c)};o.createBytes=function(a,c){for(var d=
    0,b=0,e=0,f=Array(c.length),i=Array(c.length),g=0;g<c.length;g++){var n=c[g].dataCount,h=c[g].totalCount-n,b=Math.max(b,n),e=Math.max(e,h);f[g]=Array(n);for(var k=0;k<f[g].length;k++)f[g][k]=255&a.buffer[k+d];d+=n;k=j.getErrorCorrectPolynomial(h);n=(new q(f[g],k.getLength()-1)).mod(k);i[g]=Array(k.getLength()-1);for(k=0;k<i[g].length;k++)h=k+n.getLength()-i[g].length,i[g][k]=0<=h?n.get(h):0}for(k=g=0;k<c.length;k++)g+=c[k].totalCount;d=Array(g);for(k=n=0;k<b;k++)for(g=0;g<c.length;g++)k<f[g].length&&
(d[n++]=f[g][k]);for(k=0;k<e;k++)for(g=0;g<c.length;g++)k<i[g].length&&(d[n++]=i[g][k]);return d};s=4;for(var j={PATTERN_POSITION_TABLE:[[],[6,18],[6,22],[6,26],[6,30],[6,34],[6,22,38],[6,24,42],[6,26,46],[6,28,50],[6,30,54],[6,32,58],[6,34,62],[6,26,46,66],[6,26,48,70],[6,26,50,74],[6,30,54,78],[6,30,56,82],[6,30,58,86],[6,34,62,90],[6,28,50,72,94],[6,26,50,74,98],[6,30,54,78,102],[6,28,54,80,106],[6,32,58,84,110],[6,30,58,86,114],[6,34,62,90,118],[6,26,50,74,98,122],[6,30,54,78,102,126],[6,26,52,
    78,104,130],[6,30,56,82,108,134],[6,34,60,86,112,138],[6,30,58,86,114,142],[6,34,62,90,118,146],[6,30,54,78,102,126,150],[6,24,50,76,102,128,154],[6,28,54,80,106,132,158],[6,32,58,84,110,136,162],[6,26,54,82,110,138,166],[6,30,58,86,114,142,170]],G15:1335,G18:7973,G15_MASK:21522,getBCHTypeInfo:function(a){for(var c=a<<10;0<=j.getBCHDigit(c)-j.getBCHDigit(j.G15);)c^=j.G15<<j.getBCHDigit(c)-j.getBCHDigit(j.G15);return(a<<10|c)^j.G15_MASK},getBCHTypeNumber:function(a){for(var c=a<<12;0<=j.getBCHDigit(c)-
j.getBCHDigit(j.G18);)c^=j.G18<<j.getBCHDigit(c)-j.getBCHDigit(j.G18);return a<<12|c},getBCHDigit:function(a){for(var c=0;0!=a;)c++,a>>>=1;return c},getPatternPosition:function(a){return j.PATTERN_POSITION_TABLE[a-1]},getMask:function(a,c,d){switch(a){case 0:return 0==(c+d)%2;case 1:return 0==c%2;case 2:return 0==d%3;case 3:return 0==(c+d)%3;case 4:return 0==(Math.floor(c/2)+Math.floor(d/3))%2;case 5:return 0==c*d%2+c*d%3;case 6:return 0==(c*d%2+c*d%3)%2;case 7:return 0==(c*d%3+(c+d)%2)%2;default:throw Error("bad maskPattern:"+
    a);}},getErrorCorrectPolynomial:function(a){for(var c=new q([1],0),d=0;d<a;d++)c=c.multiply(new q([1,l.gexp(d)],0));return c},getLengthInBits:function(a,c){if(1<=c&&10>c)switch(a){case 1:return 10;case 2:return 9;case s:return 8;case 8:return 8;default:throw Error("mode:"+a);}else if(27>c)switch(a){case 1:return 12;case 2:return 11;case s:return 16;case 8:return 10;default:throw Error("mode:"+a);}else if(41>c)switch(a){case 1:return 14;case 2:return 13;case s:return 16;case 8:return 12;default:throw Error("mode:"+
    a);}else throw Error("type:"+c);},getLostPoint:function(a){for(var c=a.getModuleCount(),d=0,b=0;b<c;b++)for(var e=0;e<c;e++){for(var f=0,i=a.isDark(b,e),g=-1;1>=g;g++)if(!(0>b+g||c<=b+g))for(var h=-1;1>=h;h++)0>e+h||c<=e+h||0==g&&0==h||i==a.isDark(b+g,e+h)&&f++;5<f&&(d+=3+f-5)}for(b=0;b<c-1;b++)for(e=0;e<c-1;e++)if(f=0,a.isDark(b,e)&&f++,a.isDark(b+1,e)&&f++,a.isDark(b,e+1)&&f++,a.isDark(b+1,e+1)&&f++,0==f||4==f)d+=3;for(b=0;b<c;b++)for(e=0;e<c-6;e++)a.isDark(b,e)&&!a.isDark(b,e+1)&&a.isDark(b,e+
    2)&&a.isDark(b,e+3)&&a.isDark(b,e+4)&&!a.isDark(b,e+5)&&a.isDark(b,e+6)&&(d+=40);for(e=0;e<c;e++)for(b=0;b<c-6;b++)a.isDark(b,e)&&!a.isDark(b+1,e)&&a.isDark(b+2,e)&&a.isDark(b+3,e)&&a.isDark(b+4,e)&&!a.isDark(b+5,e)&&a.isDark(b+6,e)&&(d+=40);for(e=f=0;e<c;e++)for(b=0;b<c;b++)a.isDark(b,e)&&f++;a=Math.abs(100*f/c/c-50)/5;return d+10*a}},l={glog:function(a){if(1>a)throw Error("glog("+a+")");return l.LOG_TABLE[a]},gexp:function(a){for(;0>a;)a+=255;for(;256<=a;)a-=255;return l.EXP_TABLE[a]},EXP_TABLE:Array(256),
    LOG_TABLE:Array(256)},m=0;8>m;m++)l.EXP_TABLE[m]=1<<m;for(m=8;256>m;m++)l.EXP_TABLE[m]=l.EXP_TABLE[m-4]^l.EXP_TABLE[m-5]^l.EXP_TABLE[m-6]^l.EXP_TABLE[m-8];for(m=0;255>m;m++)l.LOG_TABLE[l.EXP_TABLE[m]]=m;q.prototype={get:function(a){return this.num[a]},getLength:function(){return this.num.length},multiply:function(a){for(var c=Array(this.getLength()+a.getLength()-1),d=0;d<this.getLength();d++)for(var b=0;b<a.getLength();b++)c[d+b]^=l.gexp(l.glog(this.get(d))+l.glog(a.get(b)));return new q(c,0)},mod:function(a){if(0>
    this.getLength()-a.getLength())return this;for(var c=l.glog(this.get(0))-l.glog(a.get(0)),d=Array(this.getLength()),b=0;b<this.getLength();b++)d[b]=this.get(b);for(b=0;b<a.getLength();b++)d[b]^=l.gexp(l.glog(a.get(b))+c);return(new q(d,0)).mod(a)}};p.RS_BLOCK_TABLE=[[1,26,19],[1,26,16],[1,26,13],[1,26,9],[1,44,34],[1,44,28],[1,44,22],[1,44,16],[1,70,55],[1,70,44],[2,35,17],[2,35,13],[1,100,80],[2,50,32],[2,50,24],[4,25,9],[1,134,108],[2,67,43],[2,33,15,2,34,16],[2,33,11,2,34,12],[2,86,68],[4,43,27],
    [4,43,19],[4,43,15],[2,98,78],[4,49,31],[2,32,14,4,33,15],[4,39,13,1,40,14],[2,121,97],[2,60,38,2,61,39],[4,40,18,2,41,19],[4,40,14,2,41,15],[2,146,116],[3,58,36,2,59,37],[4,36,16,4,37,17],[4,36,12,4,37,13],[2,86,68,2,87,69],[4,69,43,1,70,44],[6,43,19,2,44,20],[6,43,15,2,44,16],[4,101,81],[1,80,50,4,81,51],[4,50,22,4,51,23],[3,36,12,8,37,13],[2,116,92,2,117,93],[6,58,36,2,59,37],[4,46,20,6,47,21],[7,42,14,4,43,15],[4,133,107],[8,59,37,1,60,38],[8,44,20,4,45,21],[12,33,11,4,34,12],[3,145,115,1,146,
        116],[4,64,40,5,65,41],[11,36,16,5,37,17],[11,36,12,5,37,13],[5,109,87,1,110,88],[5,65,41,5,66,42],[5,54,24,7,55,25],[11,36,12],[5,122,98,1,123,99],[7,73,45,3,74,46],[15,43,19,2,44,20],[3,45,15,13,46,16],[1,135,107,5,136,108],[10,74,46,1,75,47],[1,50,22,15,51,23],[2,42,14,17,43,15],[5,150,120,1,151,121],[9,69,43,4,70,44],[17,50,22,1,51,23],[2,42,14,19,43,15],[3,141,113,4,142,114],[3,70,44,11,71,45],[17,47,21,4,48,22],[9,39,13,16,40,14],[3,135,107,5,136,108],[3,67,41,13,68,42],[15,54,24,5,55,25],[15,
        43,15,10,44,16],[4,144,116,4,145,117],[17,68,42],[17,50,22,6,51,23],[19,46,16,6,47,17],[2,139,111,7,140,112],[17,74,46],[7,54,24,16,55,25],[34,37,13],[4,151,121,5,152,122],[4,75,47,14,76,48],[11,54,24,14,55,25],[16,45,15,14,46,16],[6,147,117,4,148,118],[6,73,45,14,74,46],[11,54,24,16,55,25],[30,46,16,2,47,17],[8,132,106,4,133,107],[8,75,47,13,76,48],[7,54,24,22,55,25],[22,45,15,13,46,16],[10,142,114,2,143,115],[19,74,46,4,75,47],[28,50,22,6,51,23],[33,46,16,4,47,17],[8,152,122,4,153,123],[22,73,45,
        3,74,46],[8,53,23,26,54,24],[12,45,15,28,46,16],[3,147,117,10,148,118],[3,73,45,23,74,46],[4,54,24,31,55,25],[11,45,15,31,46,16],[7,146,116,7,147,117],[21,73,45,7,74,46],[1,53,23,37,54,24],[19,45,15,26,46,16],[5,145,115,10,146,116],[19,75,47,10,76,48],[15,54,24,25,55,25],[23,45,15,25,46,16],[13,145,115,3,146,116],[2,74,46,29,75,47],[42,54,24,1,55,25],[23,45,15,28,46,16],[17,145,115],[10,74,46,23,75,47],[10,54,24,35,55,25],[19,45,15,35,46,16],[17,145,115,1,146,116],[14,74,46,21,75,47],[29,54,24,19,
        55,25],[11,45,15,46,46,16],[13,145,115,6,146,116],[14,74,46,23,75,47],[44,54,24,7,55,25],[59,46,16,1,47,17],[12,151,121,7,152,122],[12,75,47,26,76,48],[39,54,24,14,55,25],[22,45,15,41,46,16],[6,151,121,14,152,122],[6,75,47,34,76,48],[46,54,24,10,55,25],[2,45,15,64,46,16],[17,152,122,4,153,123],[29,74,46,14,75,47],[49,54,24,10,55,25],[24,45,15,46,46,16],[4,152,122,18,153,123],[13,74,46,32,75,47],[48,54,24,14,55,25],[42,45,15,32,46,16],[20,147,117,4,148,118],[40,75,47,7,76,48],[43,54,24,22,55,25],[10,
        45,15,67,46,16],[19,148,118,6,149,119],[18,75,47,31,76,48],[34,54,24,34,55,25],[20,45,15,61,46,16]];p.getRSBlocks=function(a,c){var d=p.getRsBlockTable(a,c);if(void 0==d)throw Error("bad rs block @ typeNumber:"+a+"/errorCorrectLevel:"+c);for(var b=d.length/3,e=[],f=0;f<b;f++)for(var h=d[3*f+0],g=d[3*f+1],j=d[3*f+2],l=0;l<h;l++)e.push(new p(g,j));return e};p.getRsBlockTable=function(a,c){switch(c){case 1:return p.RS_BLOCK_TABLE[4*(a-1)+0];case 0:return p.RS_BLOCK_TABLE[4*(a-1)+1];case 3:return p.RS_BLOCK_TABLE[4*
(a-1)+2];case 2:return p.RS_BLOCK_TABLE[4*(a-1)+3]}};t.prototype={get:function(a){return 1==(this.buffer[Math.floor(a/8)]>>>7-a%8&1)},put:function(a,c){for(var d=0;d<c;d++)this.putBit(1==(a>>>c-d-1&1))},getLengthInBits:function(){return this.length},putBit:function(a){var c=Math.floor(this.length/8);this.buffer.length<=c&&this.buffer.push(0);a&&(this.buffer[c]|=128>>>this.length%8);this.length++}};"string"===typeof h&&(h={text:h});h=r.extend({},{render:"canvas",width:256,height:256,typeNumber:-1,
    correctLevel:2,background:"#ffffff",foreground:"#000000"},h);return this.each(function(){var a;if("canvas"==h.render){a=new o(h.typeNumber,h.correctLevel);a.addData(h.text);a.make();var c=document.createElement("canvas");c.width=h.width;c.height=h.height;for(var d=c.getContext("2d"),b=h.width/a.getModuleCount(),e=h.height/a.getModuleCount(),f=0;f<a.getModuleCount();f++)for(var i=0;i<a.getModuleCount();i++){d.fillStyle=a.isDark(f,i)?h.foreground:h.background;var g=Math.ceil((i+1)*b)-Math.floor(i*b),
    j=Math.ceil((f+1)*b)-Math.floor(f*b);d.fillRect(Math.round(i*b),Math.round(f*e),g,j)}}else{a=new o(h.typeNumber,h.correctLevel);a.addData(h.text);a.make();c=r("<table></table>").css("width",h.width+"px").css("height",h.height+"px").css("border","0px").css("border-collapse","collapse").css("background-color",h.background);d=h.width/a.getModuleCount();b=h.height/a.getModuleCount();for(e=0;e<a.getModuleCount();e++){f=r("<tr></tr>").css("height",b+"px").appendTo(c);for(i=0;i<a.getModuleCount();i++)r("<td></td>").css("width",
    d+"px").css("background-color",a.isDark(e,i)?h.foreground:h.background).appendTo(f)}}a=c;jQuery(a).appendTo(this)})}})(jQuery);
/*
模块移动函数
主要是拖拽版的单模块内容移动效果
*/
(function ($) {
    $.fn.extend({
        "moveModule": function (options) {
            return this.each(function () {
                var defaultO = {
                    axis: "top",
                    speed: "slow",
                    type: "flow",
                    hand: false
                };
                var O = $.extend(defaultO, options);
                var speed = 100,
                    moveSpeedPx = 1;
                switch(O.type){
                    case 'flow':
                        switch (O.speed) {
                            case "slowly":
                                speed = 30;
                                moveSpeedPx = 1;
                                break;
                            case "slow":
                                speed = 20;
                                moveSpeedPx = 1;
                                break;
                            case "normal":
                                speed = 10;
                                moveSpeedPx = 2;
                                break;
                            case "quick":
                                speed = 3;
                                moveSpeedPx = 2;
                                break;
                            case "quickly":
                                speed = 3;
                                moveSpeedPx = 5;
                                break;
                        }
                        break;
                    case 'single':
                        switch (O.speed) {
                            case "slowly":
                                speed = 5000;
                                break;
                            case "slow":
                                speed = 4000;
                                break;
                            case "normal":
                                speed = 3000;
                                break;
                            case "quick":
                                speed = 2000;
                                break;
                            case "quickly":
                                speed = 1000;
                                break;
                        }
                        break;
                }
                var _this = $(this),
                    movepx = 0,
                    times = null,
                    thisPar = _this.parent(),
                    thisParH = thisPar.height(),
                    thisParW = thisPar.width(),
                    firstChild = _this.children().first(),
                    thisW = null,
                    thisH = null;
                if (O.axis == "top" || O.axis == "bottom") {
                    thisH = _this.height();
                } else if (O.axis == "left" || O.axis == "right") {
                    if (_this.hasClass('pic-text-list-module')) {
                        _this.addClass('pic-text-list-module-moveL');
                        // if(_this.hasClass('pic-text-list-module-1')){
                        var liw = firstChild.find("li").width();
                        firstChild.find("li").width(liw);
                        // firstChild.find("li").width(thisParW);
                        // }
                        var li_w = 0;
                        firstChild.find("li").each(function () {
                            li_w += $(this).outerWidth();
                        });
                        firstChild.width(li_w);
                    }
                    thisW = firstChild.width();
                    _this.width(firstChild.width() * 2 + 10);
                }
                var clone = $(firstChild.clone());
                //向上
                if (O.axis == "top") {
                    movepx = 0;
                    _this.append(clone);
                    times = setTimeout(moveT, speed);
                    _this.bind("mouseleave", function () {
                        times = setTimeout(moveT, speed);
                    });
                    _this.bind("mouseenter", function () {
                        clearTimeout(times);
                    });
                }
                //向下
                if (O.axis == "bottom") {
                    movepx = -(thisH + (thisH - thisParH));
                    _this.css({
                        "margin-top": -thisH
                    }).append(clone);
                    times = setTimeout(moveB, speed);
                    _this.bind("mouseleave", function () {
                        times = setTimeout(moveB, speed);
                    });
                    _this.bind("mouseenter", function () {
                        clearTimeout(times);
                    });
                }
                //向左
                if (O.axis == "left") {
                    movepx = 0;
                    _this.css({
                        "margin-left": 0
                    }).append(clone);
                    // times = setInterval(moveL, speed);
                    times = setTimeout(moveL, speed);
                    _this.bind("mouseleave", function () {
                        times = setTimeout(moveL, speed);
                    });
                    _this.bind("mouseenter", function () {
                        clearTimeout(times);
                    });
                }
                //向右
                if (O.axis == "right") {
                    movepx = -(thisW);
                    _this.css({
                        "margin-left": movepx
                    }).append(clone);
                    times = setTimeout(moveR, speed);
                    _this.bind("mouseleave", function () {
                        times = setTimeout(moveR, speed);
                    });
                    _this.bind("mouseenter", function () {
                        clearTimeout(times);
                    });
                }

                //下移动函数
                function moveB() {
                    clearTimeout(times);
                    if (thisH != firstChild.height()) {
                        thisH = firstChild.height();
                    }
                    var mt = parseInt(_this.css("margin-top"));
                    var itemH = firstChild.children().outerHeight();
                    if (O.type == "flow") {
                        if (mt < 0) {
                            _this.css("margin-top", movepx);
                            movepx+=moveSpeedPx;
                        } else {
                            movepx = -thisH;
                            _this.css("margin-top", movepx);
                        }
                    } else if (O.type == "single") {
                        if (mt < 0) {
                            _this.animate({
                                "margin-top": mt + itemH
                            }, 500);
                        } else {
                            _this.css("margin-top", -thisH);
                            _this.animate({
                                "margin-top": -(thisH - itemH)
                            }, 500);
                        }
                    }
                    times = setTimeout(moveB, speed);
                }

                //上移动函数
                function moveT() {
                    clearTimeout(times);
                    if (thisH != firstChild.height()) {
                        thisH = firstChild.height();
                    }
                    var itemH = firstChild.children().outerHeight();
                    var mt = Math.abs(parseInt(_this.css("margin-top")));
                    if (O.type == "single") {
                        if (mt < thisH) {
                            _this.animate({
                                "margin-top": -(mt + itemH)
                            }, 500);
                        } else {
                            _this.css("margin-top", 0);
                            _this.animate({
                                "margin-top": -(itemH)
                            }, 500);
                        }
                    } else if (O.type == "flow") {
                        if (mt < thisH) {
                            _this.css("margin-top", -movepx);
                            movepx+=moveSpeedPx;
                        } else {
                            movepx = 0;
                            _this.css("margin-top", -movepx);
                        }
                    }
                    times = setTimeout(moveT, speed);
                }

                //左移动
                function moveL() {
                    clearTimeout(times);
                    if (thisParW != _this.parent().width()) {
                        if (_this.hasClass("proListmodule_1")) {
                            thisParW = _this.parent().width();
                            firstChild.children().width(thisParW);
                            clone.remove();
                            clone = $(firstChild.clone());
                            _this.css({
                                "margin-left": 0
                            }).append(clone);
                            thisW = firstChild.width();
                            _this.width(firstChild.width() * 2);
                        }
                    }
                    var itemW = firstChild.children().outerWidth();
                    var ml = Math.abs(parseInt(_this.css("margin-left")));
                    if (O.type == "single") {
                        if (ml < thisW) {
                            _this.animate({
                                "margin-left": -(ml + itemW)
                            }, 500);
                        } else {
                            _this.css("margin-left", 0);
                            _this.animate({
                                "margin-left": -(itemW)
                            }, 500);
                        }
                    } else if (O.type == "flow") {
                        if (ml < thisW) {
                            _this.css("margin-left", -movepx);
                            if(O.speed)
                            movepx+=moveSpeedPx;
                        } else {
                            movepx = moveSpeedPx*2;
                            _this.css("margin-left", -movepx);
                        }
                    }
                    times = setTimeout(moveL, speed);
                }

                //右移动
                function moveR() {
                    clearTimeout(times);
                    if (thisParW != _this.parent().width()) {
                        if (_this.hasClass("pic-text-list-module-1") || _this.hasClass("pic-text-list-module-2")) {
                            thisParW = _this.parent().width();
                            firstChild.children().width(thisParW);
                            clone.remove();
                            clone = $(firstChild.clone());
                            _this.css({
                                "margin-left": -firstChild.width()
                            }).append(clone);
                            thisW = firstChild.width();
                            _this.width(firstChild.width() * 2);
                        }
                    }
                    var itemW = firstChild.children().outerWidth();
                    var ml = parseInt(_this.css("margin-left"));
                    if (O.type == "single") {
                        if (ml < 0) {
                            _this.animate({
                                "margin-left": ml + itemW
                            }, 500);
                        } else {
                            _this.css("margin-left", -thisW);
                            _this.animate({
                                "margin-left": -(thisW - itemW)
                            }, 500);
                        }
                    } else if (O.type == "flow") {
                        if (ml < 0) {
                            _this.css("margin-left", movepx);
                            movepx+=moveSpeedPx;
                        } else {
                            movepx = -thisW+(moveSpeedPx*2);
                            _this.css("margin-left", movepx);
                        }
                    }
                    times = setTimeout(moveR, speed);
                }

            });
        }
    });
})(jQuery);
/*
* 模块中的页码jquery插件
* */
(function($) {
    function Page(options) {
        this.config = {
            skin: 'skin-1',
            dataTotal: 5,
            pageDataNum: 5,
            curPage: 1,
            pagepageGroups: 1,
            skip: false,
            hash: false,
            callBack: null
        };
        this.config = $.extend(this.config, options);
        this.render(true);
    }
    //分页视图
    Page.prototype.view = function() {
        var that = this,
            conf = that.config,
            view = [],
            dict = {},
            hostNameHref,
            hash = location.hash,
            reg,
            search = location.search;
        if(search){
            if(search.indexOf('?p_m_page') === 0){
                reg = /\?p_m_page=(\d+)-(\d+)/g;
            }else if(search.indexOf('&p_m_page') > 0){
                reg = /&p_m_page=(\d+)-(\d+)/g;
            }
            search = search.replace(reg, '');
        }

        hostNameHref = location.pathname + search + (search ? '&' : '?');
        conf.prev = $weisiteLa.pagePrev;
        conf.next = $weisiteLa.pageNext;

        if (conf.pageSum <= 1) {
            return '';
        }
        if (conf.pageGroups > conf.pageSum) {
            conf.pageGroups = conf.pageSum;
        }

        //计算当前组
        dict.index = Math.ceil((conf.curPage + ((conf.pageGroups > 1 && conf.pageGroups !== conf.pageSum) ? 1 : 0)) / (conf.pageGroups === 0 ? 1 : conf.pageGroups));
        //当前页非首页，则输出上一页
        if (conf.curPage > 1) {
            view.push('<a href="' + hostNameHref + 'p_m_page=' + conf.mId + '-' + (conf.curPage - 1) + '#Mo_' + conf.mId + '" class="page-prev" data-page="' + (conf.curPage - 1) + '">' + conf.prev + '</a>');
        } else {
            view.push('<span class="page-prev disabled">' + conf.prev + '</span>');
        }
        //当前组非首组，则输出首页
        if (dict.index > 1 && conf.pageGroups !== 0) {
            view.push('<a href="' + hostNameHref + 'p_m_page=' + conf.mId + '-1#Mo_' + conf.mId + '"  class="page-noitem" data-page="1">1</a><b class="page-ellipsis">&#x2026;</b>');
        }
        //输出当前页组
        dict.poor = Math.floor((conf.pageGroups - 1) / 2);
        dict.start = dict.index > 1 ? conf.curPage - dict.poor : 1;
        dict.end = dict.index > 1 ? (function() {
            var max = conf.curPage + (conf.pageGroups - dict.poor - 1);
            return max > conf.pageSum ? conf.pageSum : max;
        }()) : conf.pageGroups;
        if (dict.end - dict.start < conf.pageGroups - 1) { //最后一组状态
            dict.start = dict.end - conf.pageGroups + 1;
        }
        for (; dict.start <= dict.end; dict.start++) {
            if (dict.start === conf.curPage) {
                view.push('<strong class="page-curPageent page-noitem">' + dict.start + '</strong>');
            } else {
                view.push('<a href="' + hostNameHref + 'p_m_page=' + conf.mId + '-' + dict.start + '#Mo_' + conf.mId + '" class="page-noitem" data-page="' + dict.start + '">' + dict.start + '</a>');
            }
        }

        //总页数大于连续分页数，且当前组最大页小于总页，输出尾页
        if (conf.pageSum > conf.pageGroups && dict.end < conf.pageSum && conf.pageGroups !== 0) {
            view.push('<b class="page-ellipsis">&#x2026;</b><a href="javascript:;" class="page-noitem"  data-page="' + conf.pageSum + '">' + conf.pageSum + '</a>');
        }
        //当前页不为尾页时，输出下一页
        dict.flow = !conf.prev && conf.pageGroups === 0;
        if (conf.curPage !== conf.pageSum || dict.flow) {
            view.push((function() {
                return (dict.flow && conf.curPage === conf.pageSum) ?
                    '<span class="layui-laypage-nomore" title="&#x5DF2;&#x6CA1;&#x6709;&#x66F4;&#x591A;">' + conf.next + '</span>' :
                    '<a href="' + hostNameHref + 'p_m_page=' + conf.mId + '-' + (conf.curPage + 1) + '#Mo_' + conf.mId + '"  class="page-next" data-page="' + (conf.curPage + 1) + '">' + conf.next + '</a>';
            }()));
        } else {
            view.push('<span class="page-next disabled">' + conf.next + '</span>');
        }
        view.push('<small class="page-sum">'+$weisiteLa.pageTotal.split("$")[0]+'<em>' + conf.pageSum + '</em>'+$weisiteLa.pageTotal.split("$")[1]+'</small>');
        return '<div class="inner page-' + (conf.skin ? (function(skin) {
            return /^#/.test(skin) ? 'molv' : skin;
        }(conf.skin)) : 'default') + '" id="layui-laypage-' + that.config.item + '">' + view.join('') + function() {
            return conf.skip ?
                '<div class="page-form">'+$weisiteLa.pageGo.split("$")[0]+'<input class="page-num-input" type="number" min="1" max="' + conf.pageSum + '" value="' + conf.curPage + '">'+$weisiteLa.pageGo.split("$")[1]+'<em class="page-enter-btn"><s>'+$weisiteLa.pageEnter+'</s></em></div>' : '';
        }() + '</div>';
    };

    //跳页
    Page.prototype.callBack = function(elem) {
        if (!elem) return;
        var that = this,
            conf = that.config;
        !elem.data('bindevent') && elem.on({
            click: function() {
                var $this = $(this),
                    curPage = null;
                switch (true) {
                    case ($this.hasClass('page-noitem') || $this.hasClass('page-prev') || $this.hasClass('page-next')):
                        curPage = $this.attr('data-page') | 0;
                        conf.curPage = curPage;
                        that.render();
                        break;
                    case $this.hasClass('page-enter-btn'):
                        curPage = elem.find('input.page-num-input').val().replace(/\s|\D/g, '') | 0;
                        curPage > conf.pageSum && (curPage = conf.pageSum);
                        if (curPage && curPage <= conf.pageSum) {
                            conf.curPage = curPage;
                            that.render();
                        }
                        break;
                }
                return false;
            }
        }, 'a,em').data('click', 1);
    };
    //渲染分页
    Page.prototype.render = function(load) {
        var that = this,
            conf = that.config,
            view = that.view();
        conf.eleDom.html(view);
        conf.eleDom.data('click') && conf.callBack && conf.callBack(conf, load);
        !conf.eleDom.data('click') && that.callBack(conf.eleDom);
        if (conf.hash && !load) {
            location.hash = '!' + conf.hash + '=' + conf.curPage;
        }
    };
    $.fn.extend({
        pagination: function(options) {
            options.eleDom = $(this);
            options.pageSum = Math.ceil(options.dataTotal / options.pageDataNum);
            options.curPage > options.pageSum && (options.curPage = options.pageSum);
            var page = new Page(options);
        }
    });
})(jQuery);
/*
* 关于滚动数字jquery countUp插件
* */
(function ($) {
    //自执行方法，用来兼容requestAnimationFrame方法
    (function () {
        var lastTime = 0;
        var vendors = ['webkit', 'moz', 'ms', 'o'];
        for (var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
            window.requestAnimationFrame = window[vendors[x] + 'RequestAnimationFrame'];
            window.cancelAnimationFrame = window[vendors[x] + 'CancelAnimationFrame'] || window[vendors[x] + 'CancelRequestAnimationFrame'];
        }
        if (!window.requestAnimationFrame) {
            window.requestAnimationFrame = function (callback, element) {
                var currTime = new Date().getTime();
                var timeToCall = Math.max(0, 16 - (currTime - lastTime));
                var id = window.setTimeout(function () {
                    callback(currTime + timeToCall);
                }, timeToCall);
                lastTime = currTime + timeToCall;
                return id;
            };
        }
        if (!window.cancelAnimationFrame) {
            window.cancelAnimationFrame = function (id) {
                clearTimeout(id);
            };
        }
    }());
    /*定义CountUp对象*/
    function CountUp(j_) {
        var self = this;
        // default options
        self.options = $.extend({
            target: null,
            startVal: 0,
            endVal: 100,
            decimals: 0,
            duration: 2,
            grouping: true, // 1,000,000 vs 1000000  是否开启千进制分组
            separator: ',', // character to use as a separator
            decimal: '.', // character to use as a decimal
            easingFn: 'easeOut', // optional custom easing function, default is Robert Penner's easeOutExpo
            prefix: '', // optional text before the result
            suffix: '', // optional text after the result
            numerals: [], // optionally pass an array of custom numerals for 0-9,
            startCallback: null,
            runCallback: null,
            endCallback: null
        }, j_);
    }
    CountUp.prototype = {
        constructor: CountUp,
        version: '1.0.0',
        formatNumber: function (num) {
            var self = this,
                j = self.options,
                neg = (num < 0),
                x, x1, x2, x3, i, len;
            num = Math.abs(num).toFixed(j.decimals);
            num += '';
            x = num.split('.');
            x1 = x[0];
            x2 = x.length > 1 ? j.decimal + x[1] : '';
            if (j.grouping) {
                x3 = '';
                for (i = 0, len = x1.length; i < len; ++i) {
                    if (i !== 0 && ((i % 3) === 0)) {
                        x3 = j.separator + x3;
                    }
                    x3 = x1[len - i - 1] + x3;
                }
                x1 = x3;
            }
            // optional numeral substitution
            if (j.numerals.length) {
                x1 = x1.replace(/[0-9]/g, function (w) {
                    return j.numerals[+w];
                });
                x2 = x2.replace(/[0-9]/g, function (w) {
                    return j.numerals[+w];
                });
            }
            return (neg ? '-' : '') + j.prefix + x1 + x2 + j.suffix;
        },
        // Robert Penner's easeOutExpo
        easeOut: function (t, b, c, d) {
            return c * (-Math.pow(2, -10 * t / d) + 1) * 1024 / 1023 + b;
        },
        ensureNumber: function (n) {
            return (typeof n === 'number' && !isNaN(n));
        },
        initialize: function () {
            var self = this,
                j = self.options;
            if (self.initialized) {
                return true
            }
            self.error = '';
            if (!j.target.length) {
                self.error = '[CountUp] target is null or undefined';
                return false;
            }
            if (self.ensureNumber(j.startVal) && self.ensureNumber(j.endVal)) {
                j.decimals = Math.max(0, j.decimals || 0);
                self.dec = Math.pow(10, j.decimals);
                j.duration = j.duration * 1000 || 2000;
                self.countDown = (j.startVal > j.endVal);
                self.frameVal = j.startVal;
                return self.initialized = true;
            } else {
                self.error = '[CountUp] startVal (' + j.startVal + ') or endVal (' + j.endVal + ') is not a number';
                return false;
            }
        },
        // Print value to target
        printValue: function (value) {
            var self = this,
                j = self.options;
            j.runCallback && j.runCallback(value);
            result = self.formatNumber(value);
            if (j.target[0].tagName.toLocaleLowerCase() === 'input') {
                j.target.val(result);
            } else {
                j.target.html(result);
            }
        },
        count: function (timestamp) {
            var self = this,
                j = self.options;
            if (!self.startTime) {
                self.startTime = timestamp;
            }
            self.timestamp = timestamp;
            var progress = timestamp - self.startTime;
            self.remaining = j.duration - progress;
            // to ease or not to ease
            if (j.easingFn) {
                if (self.countDown) {
                    self.frameVal = j.startVal - self[j.easingFn](progress, 0, j.startVal - j.endVal, j.duration);
                } else {
                    self.frameVal = self[j.easingFn](progress, j.startVal, j.endVal - j.startVal, j.duration);
                }
            } else {
                if (self.countDown) {
                    self.frameVal = j.startVal - ((j.startVal - j.endVal) * (progress / j.duration));
                } else {
                    self.frameVal = j.startVal + (j.endVal - j.startVal) * (progress / j.duration);
                }
            }

            // don't go past endVal since progress can exceed duration in the last frame
            if (self.countDown) {
                self.frameVal = (self.frameVal < j.endVal) ? j.endVal : self.frameVal;
            } else {
                self.frameVal = (self.frameVal > j.endVal) ? j.endVal : self.frameVal;
            }

            // decimal
            self.frameVal = Math.round(self.frameVal * self.dec) / self.dec;

            // format and print value
            self.printValue(self.frameVal);

            // whether to continue
            if (progress < j.duration) {
                self.rAF = requestAnimationFrame(function () {
                    self.count(arguments[0]);
                });
            } else {
                j.endCallback && j.endCallback();
            }
        },
        // start your animation
        start: function () {
            var self = this,
                j = self.options;
            if (!self.initialize()) {
                return;
            }
            j.startCallback && j.startCallback();
            self.rAF = requestAnimationFrame(function () {
                self.count(arguments[0]);
            });
        },
        // toggles pause/resume animation
        pauseResume: function () {
            var self = this,
                j = self.options;
            if (!self.paused) {
                self.paused = true;
                cancelAnimationFrame(self.rAF);
            } else {
                self.paused = false;
                delete self.startTime;
                j.duration = self.remaining;
                j.startVal = self.frameVal;
                requestAnimationFrame(function () {
                    self.count(arguments[0]);
                });
            }
        },
        // reset to startVal so animation can be run again
        reset: function () {
            var self = this,
                j = self.options;
            self.paused = false;
            delete self.startTime;
            self.initialized = false;
            if (self.initialize()) {
                cancelAnimationFrame(self.rAF);
                self.printValue(j.startVal);
            }
        },
        // pass a new endVal and start animation
        update: function (newEndVal) {
            var self = this,
                j = self.options;
            if (!self.initialize()){
                return;
            }
            if (!self.ensureNumber(newEndVal)) {
                self.error = '[CountUp] update() - new endVal is not a number: ' + newEndVal;
                return;
            }
            self.error = '';
            if (newEndVal === self.frameVal){
                return;
            }
            cancelAnimationFrame(self.rAF);
            self.paused = false;
            delete self.startTime;
            j.startVal = self.frameVal;
            j.endVal = newEndVal;
            self.countDown = (j.startVal > j.endVal);
            self.rAF = requestAnimationFrame(function () {
                self.count(arguments[0]);
            });
        }
    };
    $.fn.extend({
        countUp: function (j) {
            if (j.target.length) {
                this.data('countup', new CountUp(j));
            } else {
                console.log('没找到对象');
            }
        }
    });
}(jQuery));

// swf定义一个网站全局的对象，用来存储前台网站的各个方法。是webSiteFunction
wsf = {};
wsf.baiduAlter = false;
wsf.pDom = {};
wsf.f = { //功能性函数
    utf16toEntities: function (str) {
        var patt = /[\ud800-\udbff][\udc00-\udfff]/g; // 检测utf16字符正则
        str = str.replace(patt, function (char) {
          var H, L, code;
          if (char.length === 2) {
            H = char.charCodeAt(0); // 取出高位
            L = char.charCodeAt(1); // 取出低位
            code = (H - 0xd800) * 0x400 + 0x10000 + L - 0xdc00; // 转换算法
            return "&#" + code + ";";
          } else {
            return char;
          }
        });
        return str;
    },  
    s_j: function (st) { //字符串转换成json
        var j;
        if (!!st) {
            j = "{" + st + "}";
            j = eval('(' + j + ')');
            return j;
        } else {
            j = {};
            return j;
        }
    },
    j_s: function (j) { //将json转换成字符串
        var x = [];
        for (var i in j) {
            x.push(i + ":" + j[i]);
        }
        return x.join(",");
    },
    addLoadingWait: function (o) { //添加loadingWait
        o = o ? o : $('body');
        var w = o.width(),
            h = o.height();
        var loading = $('<div class="loading-wait" style="width:' + w + 'px; height:' + h + 'px;"></div>');
        o.append(loading);
    },
    alertWindow: function (text, icon, times) {
        var icons = ['warn', 'error', 'right', 'load'];
        icons = isNaN(parseInt(icon)) ? icon : icons[icon];
        return window.$.popup({
            // addTarget: parWin.Ev.pubVar.winDocum.find("body"),
            type: 1,
            animate: {
                type: 1,
                target: 1
            },
            head: {
                yes: 0
            },
            opBut: {
                yes: 0
            },
            autoClose: {
                yes: times === 'infinite' ? 0 : 1,
                time: times ? times : 1.5
            },
            but: {
                yes: 0
            },
            con: {
                text: [1, text],
                img: [1, icons]
            }
        });
    },
    confirmWindow: function (text, icon, efun, rfun) {
        return $.popup({
            type: 2,
            animate: {
                type: 1
            },
            cName: "evPopupWhite",
            area: {
                w: 300
            },
            con: {
                text: [1, text],
                img: [1, icon]
            },
            but: {
                yes: 1,
                button: {
                    but_1: {
                        text: $weisiteLa.QueDing,
                        fun: function () {
                            efun && efun();
                        }
                    },
                    but_2: {
                        text: $weisiteLa.QuXiao,
                        fun: function () {
                            rfun && rfun();
                        }
                    }
                }
            }
        });
    },
    openWin: function (tit, url, w, h) {
        w = w ? w : 600;
        h = h ? h : 500;
        var popups = $.popup({
            cName: "evPopupOpacity",
            type: 5,
            area: {
                w: w,
                h: h
            },
            head: {
                text: tit
            },
            animate: {
                type: 1
            },
            con: {
                src: url
            }
        });
        return {
            w: $("#popupIframe_" + popups),
            n: popups
        };
    },
    showWXCode: function (src) {
        var popups = $.popup({
            cName: "evPopupWhite",
            type: 4,
            area: {
                w: 'auto',
                h: 'auto'
            },
            head: {
                yes: 0
            },
            animate: {
                type: 1
            },
            con: {
                html: '<div class="show-wx-code"><div class="wx-code-img"><img src="'+src+'" /></div><p>微信扫一扫</p></div>'
            }
        });
        /*return {
            w: $("#popupIframe_" + popups),
            n: popups
        };*/
    },
    showCon: function (html) {
        var popups = $.popup({
                                 cName: "evPopupWhite",
                                 type: 4,
                                 area: {
                                     w: 'auto',
                                     h: 'auto'
                                 },
                                 head: {
                                     yes: 0
                                 },
                                 animate: {
                                     type: 1
                                 },
                                 con: {
                                     html: html
                                 }
                             });
    },
    reg: {
        phoneTelDigit: function (v) {
            var reg = /((^(\+86|86)?[0]?1[3578]\d{9})|(^\d{3,4}\-\d{7,8})|(^\d{3,4}\-\d{7,8})\-\d{1,4})$/;
            var z = reg.test(v);
            return z;
        }
    },
    tip: {
        tipDom: null,
        addTip: function(j){
            if(!j.getDom.data('tip')){
                j.getDom.data('tip',1);
                var that = this,
                    getOffset = j.getDom.offset(),
                    getWidth = j.getDom.outerWidth(),
                    getHeigght = j.getDom.outerHeight(),
                    movetop = null,
                    top = null, movetop = null;
                that.tipDom = $('<div class="tip" id="tip"><em class="arrow"></em></div>');
                j.img && that.tipDom.append('<div class="tip-img"><img src="'+j.img+'" /></div>');
                j.text && that.tipDom.append('<p class="tip-text">' + j.text + '</p>');
                j.html && that.tipDom.append(j.html);
                wsf.pDom.bodyDom.append(that.tipDom);
                that.tipDom.css({
                    display: 'block',
                    opacity: 0,
                    left: getOffset.left - (that.tipDom.outerWidth() - getWidth) / 2 + 'px'
                });
                if(getOffset.top - $(document).scrollTop() > that.tipDom.outerHeight()){
                    top = getOffset.top - that.tipDom.outerHeight();
                    movetop = -30;
                    that.tipDom.removeClass('bottom').addClass('top');
                }else{
                    top = getOffset.top + getHeigght;
                    movetop = 30;
                    that.tipDom.removeClass('top').addClass('bottom');
                }
                that.tipDom.css({
                    top: (top + movetop) + 'px'
                }).animate({
                    top: top + 'px',
                    opacity: 1
                });
            }
        },
        delTip: function(j){
            var that = this;
            that.tipDom && that.tipDom.remove();
            j.getDom.removeData('tip');
        }
    },
    upJsFun: function (curDom) {
        var getDom = null;
        switch(true){
            case curDom.hasClass('customModule'):
                getDom = curDom.find('.MoBodyC > div');
                break;
            case curDom.hasClass('ev-module-edit'):
                getDom = curDom.find('.ev-module-edit-box > div');
                break;
        }
        if(getDom && getDom.length){
            switch(true){
                // newMo
                case getDom.hasClass('ev-pic'):
                case getDom.hasClass('ev-icon-pic'):
                    wsf.f.imgLazyLoading(curDom);
                    break;
                case getDom.hasClass('ev-banner-module'):
                    wsf.banner3D.init(getDom);
                    break;
                case getDom.hasClass('ev-albums-module'):
                        wsf.textListModule(getDom);
                        wsf.albums(getDom);
                        wsf.f.imgLazyLoading(curDom);
                    break;
                case getDom.hasClass('ev-search-module'):
                    wsf.evSearch.load(getDom);
                    break;
                case getDom.hasClass('ev-audio-module'):
                    wsf.audioModule.load(getDom);
                    break;
                case getDom.hasClass('ev-custom-video'):
                    wsf.videoModule.load(getDom);
                    break;
                case getDom.hasClass('ev-sphere-module'):
                    wsf.sphereModule.load(getDom);
                    break;
                // oldmo
                case getDom.hasClass('text-list-module'):
                    setTimeout(function () {
                        wsf.textListModule(getDom);
                        wsf.moveModule(getDom);
                        wsf.f.imgLazyLoading(curDom);
                    }, 100);
                    break;
                case getDom.hasClass('pic-text-module'):
                    wsf.f.imgLazyLoading(curDom);
                    break;
                case getDom.hasClass('focus-pic-module'):
                    wsf.focusPicModule(curDom.find('.mo').attr('id'));
                    break;
                case getDom.hasClass('pic-text-list-module'):
                    setTimeout(function () {
                        wsf.textListModule(getDom);
                        wsf.moveModule(getDom);
                        wsf.sysDoc(curDom.find('.ev-sys-doc-module'));
                        wsf.f.imgLazyLoading(curDom);
                    }, 100);
                    break;
                case getDom.hasClass('form-module'):
                    if(arguments[1]){
                        wsf.formModule.submit_verify(curDom.find('form'), arguments[1]);
                        wsf.formModule.defaultValVerify(getDom);
                    }
                    break;
                case getDom.hasClass('video-module'):
                    setTimeout(function () {
                        wsf.videoModule.load(getDom);
                    }, 100);
                    break;
                case getDom.hasClass('catalogList'):
                    setTimeout(function () {
                        wsf.catalogList(getDom);
                    }, 100);
                    break;
                case getDom.hasClass('menu-catalog-module'):
                    setTimeout(function () {
                        wsf.menuCatalogModule(getDom);
                        wsf.f.imgLazyLoading(curDom);
                    }, 100);
                    break;
                case getDom.hasClass('pic-module'):
                    wsf.f.imgLazyLoading(curDom);
                    break;
            }
        }
    },
    animationLoadMo: function (dom) {
        if (dom.attr('data-loadanimate') != 1) {
            dom.attr('data-loadanimate', 1);
            var v = {
                'dom': dom,
                'st': dom.attr('style') || '',
                'aName': dom.data('animate-name'),
                'aDelay': dom.data('animate-delay'),
                'aDuration': dom.data('animate-duration')
            };
            if (v.aName) {
                v.dom.addClass('animated ' + v.aName).css({
                    'animation-delay': v.aDelay,
                    'animation-duration': v.aDuration
                });
                setTimeout(function () {
                    v.dom.removeClass('load-animate');
                }, v.aDelay.slice(0, v.aDelay.length - 1) * 1000 + 30);
                setTimeout(function () {
                    v.dom.attr('style', v.st).removeClass('animated ' + v.aName);
                    dom.hasClass('ev-module-edit-box') && (function(){
                        var videoModule = dom.find('div.ev-custom-video');
                        if(videoModule.length){
                            wsf.videoModule.load(videoModule);
                        }
                        var audioModule = dom.find('div.ev-audio-module');
                        if(videoModule.length){
                            wsf.audioModule.load(audioModule);
                        }
                        var progressModule = dom.find('div.ev-progress-module');
                        if(progressModule.length){
                            wsf.f.progressLoading(progressModule);
                        }
                    }());
                    // 关于视频的加载
                    dom.hasClass('customModule') && (function(){
                        var videoModule = dom.find('div.video-module');
                        if(videoModule.length){
                            wsf.videoModule.load(videoModule);
                        }
                    })();
                    wsf.f.imgLazyLoading(v.dom);
                    v.dom.find('.ev-module-edit-box,.customModule').each(function (i, dom) {
                        dom = $(dom);
                        dom.hasClass('ev-module-edit-box') && (function(){
                            var videoModule = dom.find('div.ev-custom-video');
                            if(videoModule.length){
                                wsf.videoModule.load(videoModule);
                            }
                            var audioModule = dom.find('div.ev-audio-module');
                            if(videoModule.length){
                                wsf.audioModule.load(audioModule);
                            }
                            var progressModule = dom.find('div.ev-progress-module');
                            if(progressModule.length){
                                wsf.f.progressLoading(progressModule);
                            }
                        }());
                        // 关于视频的加载
                        dom.hasClass('customModule') && (function(){
                            var videoModule = dom.find('div.video-module');
                            if(videoModule.length){
                                wsf.videoModule.load(videoModule);
                            }
                        })();
                        wsf.f.animationLoadMo(dom);
                    });
                }, v.aDelay.slice(0, v.aDelay.length - 1) * 1000 + v.aDuration.slice(0, v.aDuration.length - 1) * 1000 + 1);
            }
        }
    },
    imgLazyLoading: function(curDom){
        curDom.find('.lazy-loading').each(function(i, dom){
            dom = $(dom);
            var oImage = new Image();
            oImage.onload = function () {
                dom.attr('src', this.src).addClass('lazy-loading-animate');
                setTimeout(function(){
                    dom.removeClass('lazy-loading lazy-loading-animate');
                }, 1000);
            };
            oImage.src = dom.attr('data-original-src');
        });
    },
    progressLoading: function(curDom){
        wsf.progress.load(curDom.eq(0));
    },
    positionShow: function (curDom, pos) {
        // 计算模块的九个位置
        var domW = curDom.width(),
            domH = curDom.height(),
            posArray = [
                [0, 'auto', 'auto', 0, 0, 0],
                [0, 'auto', 'auto', '50%', 0, '-' + domW/2 + 'px'],
                [0, 0, 'auto', 'auto', 0,  '-' + domW],
                ['50%', 'auto', 'auto', 0, '-' + domH/2 + 'px', 0],
                ['50%', 'auto', 'auto', '50%', '-' + domH/2 + 'px',  '-' + domW/2 + 'px'],
                ['50%', 0, 'auto', 'auto', '-' + domH/2 + 'px', 0],
                ['auto', 'auto', 0, 0, 0, 0],
                ['auto', 'auto', 0, '50%', 0, '-' + domW/2 + 'px'],
                ['auto', 0, 0, 'auto', 0, 0]
            ];
        curDom.css({
            'top': posArray[pos][0],
            'right': posArray[pos][1],
            'bottom': posArray[pos][2],
            'left': posArray[pos][3],
            'margin-top': posArray[pos][4],
            'margin-left': posArray[pos][5]
        });
    },
    mCScrollbar: function (obj, j) {
        if (!obj.data('cscroll')) {
            var j_ = {
                scrollInertia: 50,
                scrollbarPosition: 'outside',
                theme: "dark-3",
                autoHideScrollbar: false,
                mouseWheel: {preventDefault: true},
                scrollButtons: {
                    enable: false
                },
                callbacks: {
                    onScrollStart: function(){

                    },
                    onScroll: function(){
                        var $this = $(this);
                        if(this.mcs.topPct === 100){
                            $.scrollify.next();
                        }else if(this.mcs.topPct === 0){
                            $.scrollify.previous();
                        }
                    },
                    onTotalScroll: function(){
                        $.scrollify.next();
                    },
                    onTotalScrollBack:function(){
                        $.scrollify.previous();
                    }
                }
            };
            if (j) {
                j_ = $.extend(j_, j);
            }
            obj.mCustomScrollbar(j_);
            obj.data('cscroll', 1);
        } else {
            obj.mCustomScrollbar("update");
        }
    }
};
//分享
wsf.share = {
    shareLayerDom :null,
    wechatCode : null,
    setwin: function (w, d) {
        var iWidth = w; //弹出窗口的宽度;
        var iHeight = d; //弹出窗口的高度;
        var iTop = (window.screen.availHeight - 30 - iHeight) / 2; //获得窗口的垂直位置;
        var iLeft = (window.screen.availWidth - 10 - iWidth) / 2; //获得窗口的水平位置;
        return {
            iTop: iTop,
            iLeft: iLeft
        }
    },
    openWindow: function(getData){
        var that = this;
        var win = window.open(getData.url,
            "newwindow",
            "width=" + getData.width + ",height=" + getData.height + ", toolbar=no, menubar=no, scrollbars=no, resizable=no, location=no, status=no,top=" + that.setwin(getData.width, getData.height).iTop + ",left=" + that.setwin(getData.width, getData.height).iLeft
        );

        if(is_auto_parts_user && (getData.type == 'weibo' || getData.type == 'qzone')){
            $.ajax({
                'url':'/dom/AutoParts/share.php',
                'type':'post',
                'data':{username:user_name,ajax:1,user_user_id:readCookie('zz_userid'),type:getData.type == 'weibo' ? 2 : 4},
                'success':function(res){
                    var newRes = $.parseJSON(res);
                    if(newRes.code == 200){
                        if(newRes.data.resCnt > 0){
                            alert('恭喜您，获得1个抵扣金。今天还可以分享'+newRes.data.resCnt+'次！');
                        }else{
                            alert('恭喜您，已分享3次，获得抽奖机会！');
                        }
                    }
                }
            })
        }
    },
    bindEvent: function(){
        var that = this;
        wsf.pDom.bodyDom.on({
            'mouseenter.share': function(){
                var $this = $(this);
                if($this.hasClass('evIcon')){
                    var type = $this.data('type'),
                        data = $this.closest('.share-list').data();
                    switch (type) {
                        case 'wechat':
                            if(!$this.data('mouseenter')){
                                $this.data('mouseenter',1);
                                var hostName = location.protocol + '//' + GData.wapDomain,
                                    geturl = (data.url && ((data.url).indexOf('vip_') != -1)) ? (data.url).replace(/vip_/, 'wap_') : data.url,
                                    div = $('<div class="twoCode"><div class="code-canvas"></div></div>');

                                if (geturl.indexOf('?') === -1) {
                                    geturl += '?';
                                }

                                if (is_auto_parts_user) {
                                    geturl += encodeURIComponent('&autoPartsShare=1|'+readCookie('zz_userid'));
                                }

                                div.find('.code-canvas').qrcode({
                                    'text': geturl.indexOf('http') != -1 ? geturl : (hostName + geturl),
                                    width: 75,
                                    height: 75
                                });
                                div.append('<p>打开微信扫一扫</p>');
                                wsf.f.tip.addTip({getDom: $this.closest('li'), html: div});
                            }
                            break;
                    }
                }
            },
            'mouseleave.share': function(){
                var $this = $(this);
                if($this.hasClass('evIcon')) {
                    var type = $this.data('type');
                    switch (type) {
                        case 'wechat':
                            if($this.data('mouseenter')){
                                wsf.f.tip.delTip({getDom:$this.closest('li')});
                                $this.removeData('mouseenter');
                            }
                            break;
                    }
                }
            },
            'click.share': function(){
                var $this = $(this);
                var type = $this.data('type'),
                    data = $this.closest('.share-list').data(),
                    url = encodeURIComponent(data.url ? (data.url.indexOf('http') != -1 ? data.url : (location.origin + data.url)) : location.href),
                    title = encodeURIComponent(data.title || document.title),
                    summary = encodeURIComponent(data.summary || $('meta[name="keywords"]').attr('content')),
                    pic = encodeURIComponent(data.pic || '');
                switch(type){
                    case 'weibo':
                        that.openWindow({
                            url : 'http://service.weibo.com/share/share.php?url='+url+'&title='+title+'&count='+summary+'&pic='+pic+'&searchPic=true',
                            width: 650,
                            height: 534,
                            type:'weibo'
                        });
                        break;
                    case 'qzone':
                        that.openWindow({
                            url : 'http://sns.qzone.qq.com/cgi-bin/qzshare/cgi_qzshare_onekey?url='+url+'&title='+title+'&site='+url+'&summary='+summary+'&desc='+title+'&pics='+pic,
                            width: 650,
                            height: 534,
                            type:'qzone'
                        });
                        break;
                    case 'qq':
                        if(is_auto_parts_user){
                            url += encodeURIComponent('&autoPartsShare=2|'+readCookie('zz_userid'));
                        }
                        that.openWindow({
                            url : 'http://connect.qq.com/widget/shareqq/index.html?url='+url+'&title='+title+'&site='+url+'&summary='+summary+'&desc='+title+'&pics='+pic,
                            width: 800,
                            height: 680,
                            type:'qq'
                        });
                        break;
                    case 'douban':
                        that.openWindow({
                            url : 'http://www.douban.com/recommend/?url='+url+'&title='+title+'&sel='+summary,
                            width: 800,
                            height: 680,
                            type:'douban'
                        });
                        break;
                    case 'baidu':
                        that.openWindow({
                            url : 'http://tieba.baidu.com/f/commit/share/openShareApi?url='+url+'&title='+title+'&desc='+summary+'&comment=&pic='+pic,
                            width: 630,
                            height: 640,
                            type:'baidu'
                        });
                        break;
                }
            }
        }, '.share-list .evIcon');
    },
    createShareLayer : function(clickDom){
        var that = this;
        that.body = $('body');
        if(!that.body.data('sharelayer')){
            var listArray = [
                ['weibo-3', 'wechat-1', 'qq-9', 'qzone-1', 'baidu-3', 'douban-2'],
                [['微博', 'weibo'], ['微信', 'wechat'], ['QQ好友', 'qq'], ['QQ空间', 'qzone'], ['百度', 'baidu'], ['豆瓣', 'douban']]
            ],liArray = [];
            $.each(listArray[0], function(i, v){
                var txtarr = listArray[1][i];
                liArray.push('<li class="' + txtarr[1] + '-li"><em class="evIcon evIcon-' + v + '" data-type="' + txtarr[1] + '"></em><b class="icon-name">' + txtarr[0] + '</b></li>');
            });
            that.shareLayerDom = $('<div class="share-layer" id="shareLayer"><div class="share-layer-inner"><em class="share-layer-arrow em-1"></em><em class="share-layer-arrow em-2"></em><ul class="share-list">' + liArray.join(' ') + '</ul></div></div>');
            that.body.data('sharelayer',1).append(that.shareLayerDom);
            that.body.on('click.hideshare',function(){
                that.shareLayerDom.length && that.shareLayerDom.hide();
            });
        }
        that.shareLayerDom.length && (function(){
            var t = clickDom.offset().top,
                l = clickDom.offset().left,
                shareList = that.shareLayerDom.find('.share-list'),
                parentLi = clickDom.closest('li');
            if(parentLi.length){
                shareList.data({
                    'title': parentLi.find('.pic-title a').text() || '',
                    'pic': parentLi.find('.pic img').length ? parentLi.find('.pic img').attr('src') : '',
                    'url':parentLi.find('.pic-title a').attr('href') || '',
                    'summary':parentLi.find('.pic-intro').text() || ''
                });
            }
            that.shareLayerDom.css({
                display: 'block',
                left: l - (that.shareLayerDom.width() / 2) + clickDom.width()/2 + 'px',
                top: t + clickDom.height() + 'px'
            });
        })();
    }
};

wsf.emjio = {
    list: [128515,128516,128513,128518,128517,129315,128514,128578,128579,128521,128522,128519,129392,128525,129321,128536,128535,128538,128537,128523,128539,128540,129322,128541,129297,129303,129325,129323,129300,129296,129320,128528,128529,128566,128566,128527,128530,128580,128556,128558,129317,128524,128532,128554,129316,128564,128567,129298,129301,129314,129326,129319,129397,129398,129396,128565,128565,129327,129312,129395,128526,129299,129488,128533,128543,128577,128558,128559,128562,128563,129402,128550,128551,128552,128560,128549,128546,128557,128561,128534,128547,128542,128531,128553,128555,128548,128545,128544,129324,128520,128127,128128,128169,129313,128121,128122,128123,128125,128126,129302,128570,128568,128569,128571,128572,128573,128576,128575,128574,128139,128075,129306,128400,128406,128076,129310,129311,129304,129305,128072,128073,128070,128405,128071,128077,128078,128074,129307,129308,128079,128588,128080,129330,129309,128591],
    popup: null,
    show: function(evObj, callBack){
        var l = evObj.offset().left;
        var t = evObj.offset().top;
        var that = this;
        if(!that.popup){
            that.popup = $('<div class="emjio-popup"><i class="close" title="关闭"></i></div>');
            var arr = [];
            $.each(that.list, function(i, v){
                arr.push('<span>'+ ("&#"+ v) +'</span>');
            });
            that.popup.append(arr.join(''));
            
            $('body').append(that.popup);
            that.popup.on('click', 'span', function(){
                var getEmjio = $(this).text();
                if(callBack){
                    callBack(getEmjio);
                }
                that.hide();
            });
            that.popup.on('click', '.close', function(){
                that.hide();
            });
        }
        that.popup.css({left: l - this.popup.width() + 'px', top: t - this.popup.height() + 'px'}).show();
        setTimeout(function(){
            $('body').one('click', function(){
                that.popup.hide();
            });
        }, 10);
    },
    hide: function(){
        this.popup.hide();
    }
};

wsf.concern = function(){
    $('body').on({
        'mouseenter.concern': function(){
            var $this = $(this),
                type = $this.attr('data-type');
            switch(type){
                case 'qq':
                case 'wechat':
                    var concernList = $this.closest('.concern-list'),
                        src = concernList.attr('data-' + type);
                    if(src){
                        img = new Image();
                        img.onload = function(){
                            wsf.f.tip.addTip({getDom: $this.closest('li'), html: '<div class="twoCode"><div class="code-img"><img src="'+src+'" /></div></div>'});
                        };
                        img.src = src;
                    }
                    break;
            }
        },
        'mouseleave.concern': function(){
            var $this = $(this);
            wsf.f.tip.delTip({getDom:$this.closest('li')});
        },
        'click.concern':function(){
            var $this = $(this),
                concernList = $this.closest('.concern-list'),
                type = $this.attr('data-type'),
                url = null;
            switch(type){
                case 'weibo':
                case 'qzone':
                case 'baidu':
                    url = concernList.attr('data-' + type);
                    url && window.open(concernList.attr('data-' + type), '_blank');
                    break;
            }
        }
    }, '.concern-list .evIcon');
};
wsf.banner3D = {
    init: function (obj) {
        if (obj) {
            $.banner3D(obj);
        } else {
            //
            $('div.ev-banner-module, .customModuleRowGroup').each(function (i, dom) {
                dom = $(dom);
                dom.is(':visible') && $.banner3D(dom);
            });
        }
    },
    canvasLoad: function (curDom) {
        canvasArea = curDom.find('.canvas-area');
        // canvas动画
        switch (curDom.attr('data-canvas-type') * 1) {
            case 0:
                canvasArea.empty();
                break;
            case 1:
                canvasArea.empty();
                canvasArea.canvasRain({
                    dir: 4,
                    height: 20,
                    count: 50
                });
                break;
            case 2:
                canvasArea.empty();
                canvasArea.html('<canvas speed="1" interaction="false" size="2" count="80" opacity="0.00001" start-color="rgba(253,252,251,.9)" end-color="rgba(251,252,253,0.3)" wind-power="0" image="false"></canvas><canvas speed="3" interaction="true" size="6" count="30" start-color="rgba(253,252,251,.9)" end-color="rgba(251,252,253,0.3)" opacity="0.00001" wind-power="2" image="false"></canvas><canvas speed="3" interaction="true" size="12" count="20" wind-power="-5" image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAAEfCAMAAADMTQtEAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAyJpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDExIDY2LjE0NTY2MSwgMjAxMi8wMi8wNi0xNDo1NjoyNyAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RSZWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZVJlZiMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENTNiAoV2luZG93cykiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6QjY0QUU3MkFEOUNEMTFFNEJFNkRGNjEwRkUwMkZEMTIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6QjY0QUU3MkJEOUNEMTFFNEJFNkRGNjEwRkUwMkZEMTIiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDpCNjRBRTcyOEQ5Q0QxMUU0QkU2REY2MTBGRTAyRkQxMiIgc3RSZWY6ZG9jdW1lbnRJRD0ieG1wLmRpZDpCNjRBRTcyOUQ5Q0QxMUU0QkU2REY2MTBGRTAyRkQxMiIvPiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/PufmcIAAAAAJUExURf7+/v////////Ps+pcAAAADdFJOU///ANfKDUEAABuGSURBVHja7F0Be+s2DgP8/3/03drEAkhKchI5dbb0u9veq+Y2YSSKBEEQ2+lf/Odr+zd84XRTgcD//8+vsQ7Y6ufr///6F2wvnH8Ecf/fx1sLbziD/xzC38P4NdbQVj92+jEVP/8g4lRb7Q4LPz7ra6zJIbz9419xEM80FsjdVL+e/mus4c768Vi7m+fXWP24Yd9bd7t9jdX92aB7rK/PGt6GvOU69z32NdbwGN79Ff4FGeK5xmqX4W/g8PVZs8wQ9+P48fnO2cbab8LiMvy4jXZyIs17OMqMav18/2us4LVYxln8PId/PvjXoiyq0+IHAqinw8q/Xn7HHfSA/oZfX2NVwemvcfZv4eb++TVWgB9uhxC3nSSeDPgaKySJnh2Shgp+jRW8/C2G+DmH5P3PNwDnaywPt/YvbvKX2y35NVZxIf7iDrc4FW2/fY0VI3mo55Kt9b0Ne2DNLbyiXI/fOCsHD92vd7/jF34nzrFMFTzcoy38Zdxw2+IXMlb23A19uGc+NzTwzB1UZ6o3H8CLGOt+2UkyePdY97rYC6/4+IvovbCn673nGOuWDN69eYO0uC+c66/K6ojA3M99UqcdQy1T7N/Qb+HswlL6BdivYDy3t07ZWVKzvzmJhi23hVP3VUEHoxBUbpn8JYy1B5xsUSiBsHDuGYzW2pFt3rc6L2Es+5KPsnmwEI2udF89pmHb4i2Lv9LO2sHk/QbkHjjEN7jUB1TZgS3c0Mir+KybH+V++bQ7O15Fv7tu3e1SGwtXNRa4bys2o7F9w9/GSprb3Sb5t8jC3WrX8FloIdXuUNVejNHPMq9FuU3C/pWF28u7QiJNqHcyZIYpxtoToJU+oLjubAFPpYhnRYb3mL3dgJCER0piiJnRig+K9+skxRPAC3j2+U0D2DMdA7Qs1F5YE+N+8u3S3WOYe1yKi8RZOehK+OjtbdjCGmtRf8duLEoZoO2xi4J/N+pyAN4tpH7FWLot2yUCAT12uEOcw3ZZY/0msm2fsTGSWsb2AiREzx/0aFOd590ZXAWimdzqtwOxxRIGnwMw7wbwrXUnDGQU5I4Ubdc21rYTHCAf8P2vz+0sMhF0xCg/ZUpSU69XNvG7jQUrigkSgedNxWyslghu1Di0QLmuaiy7+to/bh/6C+fafBAhETGVIAa8hgy921gK2eBeQ3yW8CCAj+xMw4da3nULr146Gdu7zyEagblhEs993juyb7nNfvXBouGcLV7GWAL7STG10SWp6Fb/cIyLsa24bRiC5+/NW71c/8b57kmbwhCZDmPnPovv2wlT1gT16Nm2ejFHOM9YrRcFlOpOsdB5E1QorN4VtSm04tUwxwX51InG2jMLhFw/LQw80rjOeA/aQqC5R55hf21XNdZOUha28h4S2kLnA2f1RB/pU2SUGf5fUtU90WeRrWi/F6eQF4aAmD1Rba2KkdOn7OCqt6F8sArG+0Lv024sEn2i2Frmmpi/q07v1Xba829DCv4e7sjJviqeQEZ+lCFgLHujDpiTfNZc5+0stGuv/UGKBV2sknpj2hPFXUCBi+/Rg5EM93JcyyOfZrKeeRvuMXTghujCIDnOT5TMGEvKd5Y94wLbdfE01HjizlIeFhpDhCOCVqyc+RPFftgvVtw5ASRaeiDte5Sr5ckA9dx0R2LPdv8rThPbNTF+ovBxniZTK1624FjE9cA/SkmF7vHbx5xiBTl18QnUcSl1nUoWEGa0swieQx/e1jRg7CO2iNQ9VVuA7ExZYIFkUBFERgi2s3BtiGYvJGhJGHtJQYvuKjJi1fjqSmi5H0gD2sX7pYXrw8pyUTbe8h5nwBegJDg2jQh2ChYMUMbvR6OesOgRvb6xtDbcfJqkROrUdaGEKPYDKnzMRp0L9w3BF6DlvzKWyT1ghwjGCyirWJZQUZ9AmUJ9mrFadAgrFE8W0Ik3/L+X8lovO9g+xlieokibCucLPTGN5tkEL+v+/u2TjLVjTIFxCsj9xWKBeyDPGM8JngO+Xpu4jLE2K7GaCtJ9Yc9kIry39wAVSbtCYKf0b/zNMXRjeR+wwFJ5Qdji8T60rGYHFz7cWMo9gNfwM4/ZFkCDfEL2nbCg9SqD+BuHVVTDygA9s4UKdm1cCFXoDzWWdM0gEHK7fqa5c7NJ2zTBitSKN7C0ZR3vPn/SwoZ41x0AMZybve1NTV5O3ffp2v2F91rKYBf7xoG3Q6XF3J5RzQP3fmFhRVfCuxrK5ZhoL2sHBBxXQQRH1200W7iqsUh3KkYW9mjhocq6bSvrtsZ04aLGojaL6l+gYwcMMz7Mx4Pt0YYJYrpwSZ9Fa/GVPWW3lqOYx10wBZWGsByg3YzlwhWNpQS1hLqNFnh4b2kgy0ykCcysC9+GVHJRVO2GvI20cHhv7QkPUrus5EnlwuWMlYggAmBKksu8cHhvqaAnNcgvaiQrc2qc4NoZIDivmd5A8GLhkb0lBcG20To78KrG0q0iiUfBC2JeOF7NIxxs7+U0BFf2+a81luZkkNtb2yhaDGAL938e3sLGuuphxWuFSbD2FIoLAUyfABnehGkYPHRrCWzVpAfOz0OWlyKkiDW67BDBv+N3olRr2UgNb9Diwmr/rlU/Ydt2rrRG729PHEqngVDCfsfeWr2zkqZDn5VtcZa1n892lYUN1ojNj9pZDDpQXUyBjT0VfPXIWFInhZWe8VKp+Y8cvCp0i9sa4FsBuRu857yZ4Nvq7L213MGHbKS3syi8MlAjzC5EQN+xNLqlnM7P2VkurbIju/WJVe7H7AnSQPXG0241+918H2GszbATpkC0CMmM3UIlzlRYhjYStlk+jAv8IGN5AbV2QRROgwLoQvAPzko6OU2IS84wVieDDxjrCb+p3KAWog+A5503609o1MSQMCeIBsXCHxirHalHjKV+qNtXKXBn0z030jvErQstFDHUILdi4e3GUoJBZ8N1eekcgpR28+8FwNBt94N6MQxJpF8Mt/uvWPgLY+WyuhyYivUvHRT9q1xbDiksXOtFVMquBKOU5pJ915v3OstcmKcvLKwl9b8q2DQvM4gvYe7JPFZgirqPCrG/hBbu2BYzjzAH1Bl7/I2hx3o3otchGNHnoK8TIWONQiwiVbpIq2QYj2Q5doNZJYWZs6FkgrizhOY4+lSjknD8GFy1E+kG1CjLHkbYsm/E4JX6ynjnlYMCqL03k4RYtkzatHuIDi8H3v05RPotfQSgFlnXActzYxVn35hVVS49ZReECkZvAd5OAHVQ1YVctPmuDCUQfk9VGEjN2I0anDwCNAmZgZ1VWcYWYCA1gyhI8ZHoQou9FhdZJbEqd1bltEqqkHWwzQDo/YcXFaLU5ES4ykWnAzFEpyDWyanDSliVxm4rlkffgEKVQaOneS2LVXsqDbuRnji9d4QCUuJEAR/jMsqRgbRRIRo5WdtTwIyv28I0h+y0aGn3GIX+XgJbdTOsZdpYpUcMP2woKlb5ztokrA7GVax49Gu1u7DKmVJ5Pju8Tm9AIo6vc/FItIA037JxiYNnKjqzhSI6pFxBRr2zRPpMYqDw4i2myLEh8m24amfJFa1xnyXGoDdZUe/0fIKmuZnE4QOOckfKULx2KYuRgedF877hsYmKoSimFncyTC8hvY+xcjE12+304I5oxsPiF1GqqoBYYawAqznS622hQTkVhYKxNktMLIWu/oA4pm7BjZ1uprarIstmlbEQrnLZvL6/PADLYjjWyVbDo0ZSGOQi7MvUAFk8OZeN1BevYit7ZdSyfRjvQtNWlf1EGEuLnsgOTVxBPp6esaZod9mNb34TD9OZDhjL5BT0jJk8UAyOIw4jfd8oL3tRA5nMv+nSHoRgW1ckCauuLZFGLB28ikhYzm/ic1Q2A4uWI0YkQUBfRX8HAdlgYdxUKGWg1VRJbF5mojGnZZ8oCBz+6DsLsrOYulByN3N9E7C/5WbKigyzo7iMCA/1AajuwiTDKlNGVGlOS9KqagwNd+EjqLqf+WAziNeuHVGgKi9sDIOlWjS9HWjLMXy0CNPfKZN1hKgVEBeBWPmUsQLE3i/hnYCVQn8/qr3VjmRu5dIrsVjQrvgouKkLnRJsaQv4pzpE+NV5LcOzYiHFKU80VnYLRccLLUrPsHhc6NZ+OsYamNm5AVxLoIS+OssJ9eZKB9T2SLHg6VCkYWVNyTpz7MW18zJXLPAsLVhU6JDSVbTfqsVJKt8UFlBWR1ks5LejTww3zWDDLI7dY8Gi80EiRlmxFzVHBmYzgFmPVqlWWW0aA7Q1Xd2jKH+ceq6u7ux5KYSvLbQD17squinSBmKFcQpMiwAqdENSDK85UznlAyHvS6UwYdSpCpHPeiwW4PS0OLKyOPDG5Osyd4MDHWIWfYj0SfT0iLEKDUwhz5LeAC09fl6bGUxSUnqoP1FchrDckD39Tg7mMDzZsjk0FmKetdlYG3je7f2r7up1oRuDFk9UpxoFB6SOxjpdn4TSORcZS6+3RFy0Xvo8elVCWfFd448lPWFRHCNfDewDe4KYDPbxo+A8JrYyDDAcBhtZaPrcxbdnmHx4ApaNNpQo9YmXAMxIN5I+dHiNsYQllcFjhJFotE5f1QQ+GBUyPOGc5Ba79tXLj/r38ZPPGgvsALisGmis/z02ZR7RTslPtHBLmLzGmzQF00y3qz6nkNavOYZURMuLgIpvtY/G44sgFjx3pcUThrKY/iu9Uh7376DUm4szfNnBSzyUQGLpigjeNS44DXDitVB1ENjg5xiWRYJf41v2Q3wm1mV8DU8Yy9IWD2ViYlOtMKk/TY9i+UTH68W+gVDZ7CNeQRugaK4dFRi7lRV/4yncS3OULIallfGP4Ur1E0DZDyyvwkfv3IZMilxIp9iRb1HtpOFxY2lJBwkSCANDt+7OCpIq43kuNpTecNq6jqUVfD9YegyRj5kTTioQrfuhYpQ8VxwkuENiXXcBN2Ur2BTbERTceQKDugSjkqSpfjNE0vZEWAw/6pCx7A5ipVgogk6JsCEbTvWTre1069aqO09Mq60dImmgwg4irENDZ7CNwMjKYRkDnXXID6XJOo9vFCgidH5NZ3N4p0rQ/Et0RSduOfHT4/pO9DUaPVUR+2zIYiXbawtuIrAz90vr1eUTQ1ykt7+yPWyBgckY8ofHglJmFq1W8wvrV5UE0hherGIu5/qjemIGHzOodeUhIWRYsJTEFrofzxR1iIXxgfRcrYOmI5xsPlx+qfpBuJLGPK0MAzMiC75YqPhe497OCfjHkUPtOdwEg6KIHj1N8s7d4olj1uL4SNYvn4dT6weIOMpoRm+AXkIoNFeDDV/CHkDGBYOyjsiICDdDB8+ZbokvMEMooylTjxpryuHpzwLYmPR8bdJvXnDtySlowcAXY/hReYGFo8VsdPkjFC+RrmVP16I3/ptUUWD5k6sFdxeOtbZYcyYwWAhPyxMvlMI6Tqk35IsD1mz0WyXIXixgLCJCLxTAEenOwv4i08IyY+2BUy8KxzCzUvXt3ugNq2GTQ2ULWvO6l3MPLPyK6fnC2Dk+yrS8e66tm6z1iWOMPTkdGjtC6NAZA4kqvUEv7ykXwIeUbB6mpQ6VGiaURIoeSicDq95ObJlVBDVMMmRo139wYbWxUvQV3f/w8pUXiV6W6MiowFo7DpSLYShT6YcXXuQ6POTOOD34iW1Um7xImVTlQSVhTULr1YU3GYsmBT3fWV0osFfvi7W1Dirz2kLufDQPjEWmosDJPLCzurEmQfRcv20EBVS6C3z0iVD1cLeCVduK2lzKfpne+8G61daaoYAe2Ndb2PjoE/sLQ74ksWZfGW2ZI/ELw2L6MjWdZoAQrhajPWwhJcrTJ+qF2/lcYizVJB3O0qXf+nXsOmqcG7mf288rETy86uJWGSslFEO9HpNr6J/p7rOk03YYFYJYzXxMT3R/VGfh57ViVczANLeyzywCZpT/A6gVOmxnlcaNr5GVMPbBhSXGohO6RxPmaeX9R42l7s7ZNRGvLRAEL7PQwTNEyZti4Z+thUUeC8mNdtiJ1sL6QIKwOU+8D9TpsNqKALZ3VlnvpjT0xd/RgkcsCRtUdmKv0PTfL0ZKCyMCMubw77Ga4aDyiD4BDAs8lrXxNsJAb2eNSyh9I6JLYahfDzs7yzVt4AsIGLcOP1+xs7zSPmqbV8xl1IWKAw5+djdnEmy1EGSCEmkVFsYv2Vnak4JRJmPzQ9FJhtC7DY2QNeN5AUVtMi2AsSNKOXTelbXGZ4UxjV0OuzFreplylwlibdzjnVVkXUbrSb0drVOZDC2Sqsy/4hiyJKzX7fcqflAl0INGFO0VnQnkAegw1VyZiT4qtw5T9q28xmcVChroUIhRgaSmiFHr5sbzMttZkdRlKiUxD8rtDCTzEwviLJMNhU/tGnhojMlDqLh+JbWnLosAhbFytGrgdTnkSJ5YEZQyzJkF0OF/WBeJR2qu4VWQv+gzakbFOlTpqSwALPlCrChB8qPWJNJR+aUU3yZQCZxK0V8SIZSyZ20+4kgfo8ZrfQJeLB1UonRpYYmxJHaXxvtAX98cA0fw+YWUSL77m7G2QchbSw5Sfwe3+lXVCdZKWJnWWRN3CaypgBbg0/YgVSsWkb9BayjuQGUWdndTjVC+l4U3FCxsgA6RCs/Bv9+N5VPoYqk6cNJNc2lgrG4lICnDhmLtu4xVUKSCsP0WRNtSWQWlFH6UJu1LgYcRM4XPcrZpPIeHpm+tK4WJ/y6Ukbcs04RA5u4spCyuwlhDplVrZNsPTzyAA3PdFhZZDW2MLYOxDYo2U2yw0Fye/dd1eQnWY0vWXIzk4aWv403GqkEl9ICiBxYsyUalfyTUNWtOeQiRmz9xorGAtNmM6nB04d5naEXWImz3Ab/Hh2Hx8BOnTSgva5cotlHUFCwWkL7NslgdJmMdn7Z88InzxrmrFlCYC1kuiHMqejI22pzJrZc3ChTJgwJjJ/KznrCWayKlKVapI6l6QimBlWK/Z3I8wuN7/OtEY21as3QNKZ2UWTT4MT1hUwPgWrzq7BTKWz8H60xjtdqXC+LJKPZORavzRNuV7m/gnD7OGBfXNNbm+YxOoMHwLVl82p6gwXqqRwP/HZ+4s0Jvt8qMtTc/rDvYE5oashhgG9XM8GHGMjRZy3GzsUvVE2GmPVwVtFjYLmqs/vvOTU7H2peqBnvQxFecJ54XrmusmYRR7A2Zi4iEJ5j5FMgtX6pEeFVjdamPQevvgVmgNmxVenpN9BuKyoaF7ZrGcpZDVkswMOpwcB2eiHzlqFYUF3hVY1n+6jwVRCQBh5WY7AkU4zOt9BYW3jyy78HaoWuQ3kptrn1nGfJBa3kpXWE+RBhbFnhhY9X01a0YkVALaEzxah9nJd5przCqEsE5s8qx6BAi6qbRZMY1WJKO9GNeni5YLry5UHHlWJXkIsbSzY8Il1QyPOBMgyB8ErRm17UDBN5qrNAzCibUAF5qtmT3EIqpo1rBCSf6ysaiDw0isvbmLrrCvHA/P6MgHnk667sNtcxYPhQlCgRDezqKEKJ5uHGahDDOe9s+0Fhh+IkOkjaQlwnjYtRkLpFDE/OfqHB9gLEiwcrGQSchi7igcEGIr+g8oVbr+mhjBeVBV2JNA//CeFoLZhWDhzI9TEb2w421x/Ghu56lEiFKGXQd4c7UUbF6AN9fg39Moq4df4RY6q+eyAuTueefZawEJPfGhNPVOcsnKrrIkeG4n2SsJDRc1i8Lyab8BKV15SG1is8xFoE8QqukatOnWaQnYPizi0rxk4zVnxCK0BvQUUXMyn4oIlrBEt2NrRxrdb6xeucLSXh7KjvjHTORBLFFTCGOZ+O7IokXdhasguqBt08YqE+O0ozA+gkkCoTCyEBsSuIljWWFBCeOVkMVxsrsgdiBzigZsyNgA+GPqJr+lbGkhFKzruJQFPabmRnFIHszdYIOTVReBM9OhZ40FrT+LnNlAKfTIkTn0VhIQ3lHQ17cmIi8JOCUauGrxmJI35hePazwXvbU6btODBh2cTNXj9EoIykEX8RYMH0+rWzqZFdhgnaE7A163nfp5EphENZEWLiWscI1H1tRdGwx++MrWUDRnEghto0KndFmCxc1VruU0o0mHCr0NChTN8aoiEXXh4cPDpzqkPyZsSIeoDx9mZtswVSnLVpnEIxkRFwRwUthYFi4ZOhQEFzaJglBRl1i8ClNgzDJd08cg4uh7uR10h0gD+4ydeWQvFRx6SC8iBEw8vjJNNbtgsYSQ6DovxehPHBAyg9SCpxhijpmV/VsZQHXNJb3ahkmQCcUdzVXghp/X94ICBBN+oAwa8y/Ap4Vhg3GLuQwym4UiEyvE+0wTpfBwalkf20sKd8jzbewOs4giBqgG9Ynq+MO3owALtN1sAwNhZI/8NRcdTIOyowI6gcZyyjEJmiX74HHkxGVYkaYY4Qz08BTjBX4WKXAsl9Zj8Zzeu58aOnbz+GC27CgjDKPvHkKmrPwDdogdTrCsN5Yhsu4W6EPUyKeqWEl7CfrPH6MsVSTTdtFgDi9EU82HtGEuu89rSLf8FY3/woGjzDWBtx8EyS0/ZVfsxP9TFEec5GoC4B/Piu6UYb19gpb8CWnKI1L2rGpAMjpZ/J5PCuyhtpIrmLo7fNYgCbiISfw7hMWYz6vgWdFSFk+9RmK9+QdkkXmjJPrINF1jNUZvpXRzyXWKmTTbF9DGwzxbJ5wlrHyKIM8IalQqnsxlMta0z6ZOybzVzGWNh7lsa1tYRVmmccsilZ/mq0Qc60/NVZkGVQ6AzsIsS3aWlHQrgmwmVyzLFzEWFWdvrjs1/WKVJxvF9kOFfHr9O40BktHrlaz6jNesxNSVDpaS4tXOYaqPB+1GVsOdBI9L4jGxlkK53KPnvRZYahbNhbO4RS7PhsSU+LA+Mv3HkPEgamItxfOgsOLiWup9RNet7zCMYSexDJx5Bn7CkFCJM9a5Yn83Mc/gfAJFoftrAY3BpVzldpP4mXXieAPjEjeTrIVXJQ4TMtorvQyuSGsz7ccu3eKsQSQ93HkGV29irE2mw9YV03PCq9sMi16pPHr4VnnSFSNjWUtmorPXNhYDaV5W3XFh2Za3+z7thZeePnb+760tc50Nt9abH1zTfdpY2kO9XuBPKpr898xlmD7bSYc+Aa23ycaa3MNDM+p8Sbf+UHGSnFwwW/6GsvvQ5mRiTdHMJ9jrC3louwJaHyNtaUZZCoh+ZYC/gcZa0vKiiKo/JbY9JOMtSEPLgrT67/G0tA0B1+Tief/WWNtWQldO39OD+U/y1gl+MGg+/M1Vn0qrTHsfM2CzzZWuwpPUlP+9xgLLr3I7zGc+ixXNv8aa+zg1V74GqtrLJgc8XdnzXEIqD7Z11iDrQUTMv1G8MNk0UZXfBPp2dbSzqevsSZuS7g0X2MdyXku3btzvTvxi8Ff7j75muD41/8EGABcagrVPWmXtgAAAABJRU5ErkJggg=="></canvas>');
                canvasArea.find('canvas').attr({
                    'height': canvasArea.height()
                }).canvasSnow();
                break;
            case 3:
                canvasArea.empty();
                canvasArea.canvasFireworks({
                    width: canvasArea.width(),
                    height: canvasArea.height()
                });
                break;
            case 4:
                canvasArea.empty();
                canvasArea.canvasMeteor({
                    width: canvasArea.width() < 810 ? 810 : canvasArea.width(),
                    height: canvasArea.height() || 100
                });
                break;
            case 5:
                canvasArea.empty();
                canvasArea.canvasAirBubble({
                    radius: 45,
                    densety: .1,
                    color: 'rgba(255,255,255, .4)',
                    //color: 'random',
                    clearOffset: .3
                });
                break;
        }
    }
};
wsf.rowVideo = function () {
    var rowvideobg = $('div.rowVideoBg');
    rowvideobg.each(function (i, dom) {
        dom = $(dom);
        var video = dom.find('video');
        video.length && (function () {
            
            setTimeout(function(){
                var vow = video.width(),
                    voh = video.height(),
                    vnh = null,
                    vnw = null,
                    vph = video.parent().height(),
                    vpw = video.parent().width();
                if (vow == vpw) {
                    if (voh < vph) {
                        video.css({height: '100%', width: 'auto'});
                        vnw = video.width();
                        vnw > vpw && video.css({left: -(vnw - vpw) / 2 + 'px'});
                    }
                    voh > vph && video.css({top: -(voh - vph) / 2 + 'px'});
                }
                if(!video[0].paused){
                    dom.siblings('.play-btn').addClass('evIcon-pause-8').removeClass('off');
                }
            }, 900);
        })();
    });
};
// 动画加载效果
wsf.loadAnimate = {
    getDom: [],
    getLoadImg: [],
    getProgress: [],
    loadFun: function(){
        var bodyDom = $('body');
        var bodyHeight = bodyDom.height(),
            that = this;
        /*延迟1.5秒后从新判断一下页面高度和实际页面高度是否相同，如果不同从新计算模块位置*/
        setTimeout(function(){
            if(bodyDom.height() !== bodyHeight){
                $.each(that.getDom, function(i, v){
                    v.oTop = v.dom.offset().top;
                });
                $.each(that.getLoadImg, function(i, v){
                    v.oTop = (function () {
                        var picTextListModule = v.dom.closest('.pic-text-list-module'),
                            top = v.dom.offset().top;
                        if (picTextListModule.length && picTextListModule.attr('data-move') * 1 === 1 && (picTextListModule.attr('data-axis') === 'top' || picTextListModule.attr('data-axis') === 'bottom')) {
                            top = picTextListModule.parent().offset().top;
                        }
                        return top;
                    })();
                });
            }
        }, 1500);
        $("div.load-animate[data-animate-name][data-animate-delay][data-animate-duration]").each(function (i, dom) {
            dom = $(dom);
            var tag = true,
                evContainer = dom.closest('.ev-container');
            if (dom.is(':visible')) {
                if (evContainer.length && evContainer.css('visibility') == 'hidden') {
                    // if (evContainer.hasClass('ev-tab-container-two')) {
                    // }
                    tag = false;
                }
            } else {
                tag = false;
            }
            tag && that.getDom.push({
                'dom': dom,
                'st': dom.attr('style') || '',
                'oTop': dom.offset().top,
                'loaded': false,
                'aName': dom.data('animate-name'),
                'aDelay': dom.data('animate-delay'),
                'aDuration': dom.data('animate-duration')
            });
        });
        $("img.lazy-loading").each(function (i, dom) {
            dom = $(dom);
            var tag = true,
                evContainer = dom.closest('.ev-container'),
                row = dom.closest('.customModuleRow');
            if (evContainer.length) {
                if (evContainer.hasClass('ev-tab-container-two') && !evContainer.hasClass('ev-tab-active')) {
                    tag = false;
                }
            }
            if(row.length){
                if (row.is(':hidden')) {
                    tag = false;
                }
            }
            tag && that.getLoadImg.push({
                'dom': dom,
                'st': dom.attr('style') || '',
                'oTop': (function () {
                    var picTextListModule = dom.closest('.pic-text-list-module'),
                        top = dom.offset().top;
                    if (picTextListModule.length && picTextListModule.attr('data-move') * 1 === 1 && (picTextListModule.attr('data-axis') === 'top' || picTextListModule.attr('data-axis') === 'bottom')) {
                        top = picTextListModule.parent().offset().top;
                    }
                    return top;
                })(),
                'loaded': false,
                'src': dom.data('original-src')
            });
        });
    },
    showAnimateMo: function(s, h){
        var that = this,
            animateTag = (s !== undefined && h != undefined);
        if (that.getDom.length > 0) {
            $.map(that.getDom, function (v, i) {
                if(animateTag){
                    if (!v.loaded && (s + h > v.oTop) && (v.oTop + v.dom.height() > s ||  (v.dom.data('fixed') || v.dom.parent().data('fixed')) * 1 === 1)) {
                        if ($.browser && $.browser.msie && $.browser.msie < 9) {
                            v.dom.removeClass('load-animate');
                            v.loaded = true;
                        } else {
                            wsf.f.animationLoadMo(v.dom);
                            v.loaded = true;
                            if (v.dom.parent().hasClass('ev-tab-container-two')) {
                                v.dom.attr('data-loadanimate', 1).data('loadanimate', 1);
                            }
                        }
                    }
                }else{
                    wsf.f.animationLoadMo(v.dom);
                    v.loaded = true;
                }
            });
        }
        if (that.getLoadImg.length > 0) {
            $.map(that.getLoadImg, function (v, i) {
                if(animateTag){
                    if (!v.loaded && (s + h > v.oTop) && (v.oTop + v.dom.height() > s || v.dom.closest('div[data-fixed="1"]').data('fixed'))) {
                        wsf.f.imgLazyLoading(v.dom.parent());
                        v.loaded = true;
                    }
                }else{

                    wsf.f.imgLazyLoading(v.dom.parent());
                    v.loaded = true;
                }
            });
        }

    }
}
//滚动的时候的固定定位模块方法
wsf.scrollFixed = {
    judge: function () {
        var that = this;
        that.fixedDomJson = {
            'wrap': $("#wrapper"),
            'alone': {},
            'group': {}
        };
        (function () {
            var domArray = [$('#top_area'), $('#web_nav'), $('#header')],
                i = 0;
            while (i < domArray.length) {
                if (domArray[i].length) {
                    var dom = domArray[i];
                    if (dom.attr('data-fixed') === '1') {
                        that.fixedDomJson.alone[dom.attr('id')] = {
                            dom: dom,
                            iDom: dom.children(["class*='Inner'"]).eq(0),
                            iDomS: dom.children(["class*='Inner'"]).eq(0).attr('style'),
                            fixed: dom.attr('data-fixed'),
                            fixedX: dom.attr('data-fixedx') == '-1' ? dom.offset().top : dom.attr('data-fixedx'),
                            oLeft: dom.offset().left,
                            oTop: dom.offset().top,
                            position: dom.css('position'),
                            width: dom.width(),
                            height: dom.height(),
                            pLeft: dom.position().left,
                            pTop: dom.position().top
                        };
                    }
                }
                i++;
            }
        })();
        $('div.headPublicModuleRow').find('.ev-module-edit,.customModule').each(function (i, dom) {
            // console.log(dom, i);
            dom = $(dom);
            var id = dom.attr('id');
            if (dom.hasClass('customModule')) {
                id = dom.find('.Mo').attr('id');
            }
            if (dom.attr('data-fixed') === '1') {
                that.fixedDomJson.group[id] = {
                    dom: dom,
                    fixed: dom.attr('data-fixed'),
                    fixedX: dom.attr('data-fixedx') == '-1' ? dom.offset().top : dom.attr('data-fixedx'),
                    oLeft: dom.offset().left,
                    oTop: dom.offset().top,
                    pLeft: (that.fixedDomJson.group[id] && that.fixedDomJson.group[id].pLeft) ? that.fixedDomJson.group[id].pLeft : dom.position().left,
                    pTop: (that.fixedDomJson.group[id] && that.fixedDomJson.group[id].pTop) ? that.fixedDomJson.group[id].pTop : dom.position().top,
                };
            }
        });
        $('#absolute_module_wrap').find('.absolute-module').each(function (i, dom) {
            dom = $(dom);
            if (dom.attr('data-fixed') === '1') {
                that.fixedDomJson.group[dom.attr('id')] = {
                    'dom': dom,
                    'fixed': dom.attr('data-fixed'),
                    'fixedX': dom.attr('data-fixedx') == '-1' ? dom.offset().top : dom.attr('data-fixedx'),
                    'oLeft': dom.offset().left,
                    'oTop': dom.offset().top,
                    'pLeft': dom.position().left,
                    'pTop': dom.position().top
                };
            }
        });
    },
    fun: function (x) {
        var alone = this.fixedDomJson.alone,
            group = this.fixedDomJson.group;
        $.each(alone, function (i, v) {
            if (v.fixed) {
                if (x >= v.fixedX) {
                    v.dom.height(v.height);
                    if (v.dom.is($('#web_nav')) && (v.width < v.iDom.width())) {
                        v.iDom.css({
                            'width': '100%',
                            'left': 0
                        });
                    } else {
                        v.iDom.css({
                            'width': v.width,
                            'left': v.oLeft
                        });
                    }
                    v.iDom.addClass('scrollFixed').css({
                        'position': 'fixed',
                        top: (v.oTop - v.fixedX) <= 0 ? 0 : (v.oTop - v.fixedX)
                    });
                } else {
                    v.iDom.removeClass('scrollFixed').attr('style', v.iDomS || '');
                }
            }
        });
        $.each(group, function (i, v) {
            if (v.fixed) {
                if (x >= v.fixedX) {

                    v.dom.css({
                        position: 'fixed',
                        left: v.oLeft,
                        top: (v.oTop - v.fixedX) < 0 ? 0 : (v.oTop - v.fixedX)
                    });
                } else {
                    v.dom.css({
                        position: 'absolute',
                        left: v.pLeft,
                        top: v.pTop
                    });
                }
            }
        });
    }
};
//给window对象绑定事件
wsf.wScroll = function (a) {
        var curWin = isParentWindow ? window.top : window,
            curWinDom = $(curWin),
            eventList = curWinDom.data('events') || $._data(curWin, 'events');
        if (isParentWindow || curWin === window) {
            if (eventList && eventList['scroll']) {
                $.each(eventList['scroll'], function (i, v) {
                    if (v.namespace === 'moCount') {
                        curWinDom.off('scroll.moCount');
                        return false;
                    }
                });
            }
        }
        if (isParentWindow) {
            var adminWrapper = $('#adminWrapper', curWin.document);
            curWinDom.on({
                'scroll.moCount': function () {
                    var st = curWinDom.scrollTop(),
                        h = curWinDom.height() - parseInt(adminWrapper.css('padding-top'));
                    wsf.loadAnimate.showAnimateMo(st, h);
                    wsf.progress.showProgress(st, h);
                }
            });
            var defst = curWinDom.scrollTop(),
                defh = curWinDom.height() - parseInt(adminWrapper.css('padding-top'));
            wsf.loadAnimate.showAnimateMo(defst, defh);
            wsf.progress.showProgress(defst, defh);
        } else {
            curWinDom.on({
                'scroll.moCount': function (event) {
                    var st = curWinDom.scrollTop(),
                        h = curWinDom[0].innerHeight;
                    wsf.loadAnimate.showAnimateMo(st, h);
                    wsf.progress.showProgress(st, h);
                    wsf.fixedRow.showFixedDom(st, h);
                    wsf.customerService.showToTop(st, h);
                    wsf.scrollFixed.fun(st);
                    // console.log(88888888)
                },
                'resize.moCount': function () {
                    wsf.fullRow.countMoPosition();
                    // var st = curWinDom.scrollTop();
                    // setTimeout(function(){
                    //     console.log(8888);
                    //     curWinDom.trigger('scroll');
                    // }, 100);
                    // console.log(st, 'st');
                    // if(window.resizeEvt){
                    //     return;
                    // }
                    // window.resizeEvt = true;
                    // setTimeout(function(){
                    //     window.location.reload();
                    //     wsf.scrollFixed.judge();
                    //     setTimeout(function() {
                    //         wsf.scrollFixed.fun(st);
                    //         window.resizeEvt = false;
                    //     }, 1000);
                    // }, 1000);
                    // wsf.fullRow.evRFSwidthmodule.length && wsf.f.positionShow(wsf.fullRow.evRFSwidthmodule, wsf.fullRow.evRFSwidthmodule.attr('data-position') - 1);
                }
            });
            var defst = curWinDom.scrollTop(),
                defh = curWinDom.height();
            wsf.loadAnimate.showAnimateMo(defst, defh);
            wsf.fixedRow.showFixedDom(defst, defh);
            wsf.progress.showProgress(defst, defh);
        }
};
// 站点导航的方法,主要作用是显示和隐藏子导航菜单。
wsf.nav = function () {
    var webNav = $('#web_nav'),
        navInner = webNav.find('.navInner'),
        fullSubNavBox = $('#fullSubNavBox'),
        documents = $(document),
        leaveJson = {
            curSubRow: null,
            hideTime: null,
            curItem: null,
            startTimeFn: function(){
                var that = this;
                that.hideTime = setTimeout(function(){
                    leaveJson.hideFun();
                }, 200);
            },
            endTimeFn: function(){
                var that = this;
                clearTimeout(that.hideTime);
            },
            hideFun: function(){
                var that = this;
                that.curItem && that.curItem.removeClass('NItemH');
                if(that.curSubRow && that.curSubRow.length){
                    that.curSubRow.removeClass('customModuleRowBlock animated').addClass('customModuleRowNone');
                    that.curSubRow = null;
                }
            },
            showFun: function(an){
                var that = this;
                if(that.curSubRow && that.curSubRow.length){
                    if(!that.curSubRow.data('ajaxload')){
                        that.curSubRow.find('.ev-module-edit,.customModule').each(function (i, dom) {
                            dom = $(dom);
                            wsf.f.upJsFun(dom);
                            if(dom.hasClass('customModule')){
                                wsf.f.animationLoadMo(dom);
                            }else{
                                var curDomEditBox = dom.find('div.ev-module-edit-box:first');
                                if(curDomEditBox.data('animate-name')){
                                    wsf.f.animationLoadMo(curDomEditBox);
                                }
                            }
                            wsf.f.imgLazyLoading(dom);
                        });
                        that.curSubRow.data('ajaxload', 1);
                        wsf.queryMJsEffect(MJsData);
                    }
                    that.curSubRow.removeClass('customModuleRowNone').addClass('customModuleRowBlock');
                    that.curItem && that.curItem.addClass('NItemH');
                    if(an){
                        setTimeout(function(){
                            that.curSubRow.addClass('animated');
                        });
                    }else{
                        that.curSubRow.addClass('animated');
                    }
                }
            }
        },
        NSubShowFun = function(curItem){
            var sub = curItem.find("div.NSub");
                if (sub.length) {
                    var l = curItem.offset().left,
                        pl = sub.width() + l;
                    if (pl > documents.width()) {
                        sub.css({
                            left: 'auto',
                            right: 0
                        });
                    }
                    curItem.addClass('NItemSub');
                }
        };
    $("#nav").on({
        mouseenter: function (event) {
            var $this = $(this),
                subLayout = webNav.attr('data-sublayout') * 1,
                animated = true;
            if (!$this.hasClass('NItemCur') && !$this.hasClass("NItemH")) {
                $this.addClass("NItemH");
            }
            switch(subLayout){
                case 2:
                    !is_action && (function(){
                        var thisSubId = $this.data('subid');
                        if(leaveJson.curSubRow){
                            animated = false;
                            clearTimeout(leaveJson.hideTime);
                            leaveJson.hideFun();
                            leaveJson.hideTime = null;
                        }
                        if(thisSubId){
                            var curSubRow = leaveJson.curSubRow = $('#row_' + thisSubId);
                            leaveJson.curItem = $this;
                            if(!curSubRow.hasClass('customModuleRowBlock')){
                                if(navInner.css('position') === 'fixed'){
                                    curSubRow.css({'position': 'fixed', top: $this.offset().top - $(document).scrollTop() + $this.height()});
                                }else{
                                    curSubRow.css({'position': 'absolute', 'top': $this.offset().top + $this.height()});
                                }
                                leaveJson.showFun(animated);
                            }
                        }else if($this.hasClass('NLast')){
                            NSubShowFun($this);
                        }
                    })();
                    break;
                default:
                    NSubShowFun($this);
                    break;
            }
        },
        mouseleave: function () {
            var $this = $(this),
                subLayout = webNav.attr('data-sublayout') * 1;
            switch(subLayout){
                case 2:
                    !is_action && $this.data('subid') && leaveJson.startTimeFn();
                    if($this.hasClass('NLast')){
                        $this.removeClass("NItemH NItemSub");
                    }else if(!$this.data('subid') || is_action){
                        $this.removeClass("NItemH");
                    }
                    break;
                default:
                    $this.removeClass("NItemSub NItemH");
                    break;
            }

        }
    }, '.NItem');
    // 给全屏子菜单绑定事件
    if(!is_action && fullSubNavBox.length){
        // 移除空的子菜单行
        fullSubNavBox.children('.customModuleRow').each(function(i, dom){
            dom = $(dom);
            var tag = false;
            if(dom.find('.ev-module-edit,.customModule').length){
                tag = true;
            }
            if(!tag){
                dom.remove();
            }
        });
        // 给子菜单行绑定事件
        fullSubNavBox.on({
            mouseenter: function(){
                if(leaveJson.curSubRow){
                    leaveJson.endTimeFn();
                }
            },
            mouseleave: function(){
                if(leaveJson.curSubRow && !leaveJson.curSubRow.hasClass('curEditRow')){
                    leaveJson.startTimeFn();
                }
            }
        }, '.customModuleRow');
    }
};
// 站点搜索框方法,主要作用是显示和隐藏默认文字。
wsf.search = function () {
    var searchModule = $("form.search-form");
    if (searchModule.length) {
        searchModule.each(function (i, formDom) {
            formDom = $(formDom);
            var formDomId = formDom.attr('id'),
                parentSearch = formDom.closest('.search');
            formDom.on('focus blur keydown', '.keyWord .input', function (event) {
                var curDom = $(this),
                    curDomVal = curDom.val(),
                    curDomDefaultVal = curDom.attr('data-defaultv');
                switch (event.type) {
                    case 'focusin':
                        curDomVal == curDomDefaultVal && curDom.val('');
                        break;
                    case 'focusout':
                        (curDomVal == curDomDefaultVal || !curDomVal) && curDom.val(curDomDefaultVal);
                        break;
                    case 'keydown':
                        event.keyCode === 13 && formDom.find('.keyBtn').trigger('click');
                        break;
                }
            });
            formDom.on({
                click: function () {
                    var keyWord = formDom.find('.keyWord .input'),
                        curDomVal = keyWord.val(),
                        curDomDefaultVal = keyWord.attr('data-defaultv');
                    if (!curDomVal || (curDomVal == curDomDefaultVal && !(keyWord.attr('data-isdefaultsearch') * 1))) {
                        wsf.f.alertWindow(parentSearch.attr('data-tishi'), 0);
                        keyWord.trigger('focus');
                        return false;
                    }
                    if (parentSearch.attr('data-search-type') == 2) {
                        formDom.submit();
                    } else {
                        window.location.href = formDom.attr("action") + "&keyWord=" + keyWord.val();
                    }
                }
            }, '.keyBtn');
            switch (formDomId) {
                case 'pageMySearch':
                    formDom.on({
                        change: function () {
                            formDom.find('.keyBtn').trigger('click');
                        }
                    }, 'select[name="navtype"]');
                    break;
            }
        });
    }
};
/*网站的logo search shopCart 位置计算*/
wsf.hoffL = function () {
    var j = 0;
    var p = function (obj, t) {
        if (t == 'n') {
            if (obj.attr('data-l')) {
                j = wsf.f.s_j(obj.attr('data-l'));
            }
        } else {
            if (obj.attr('data-s')) {
                j = wsf.f.s_j(obj.attr('data-s'));
            }
        }
    };
    var webNav = $('#web_nav');
    var hcL = Math.floor(($("body").width() - userSiteWidth) / 2);
    if (webNav.length) {
        p(webNav, 'n');
        if (j.p == 1 || j.p == 3) {
            webNav.css({
                'left': j.l + hcL
            });
        }
    }
};
/*固定定位行fixedRow*/
wsf.fixedRow = {
    fixedDom: null,
    getDom: function () {
        var that = this;
        that.fixedDom = $('div.fixedPublicModuleRow');
        if(that.fixedDom.length){
            wsf.pDom.htmlDom.on({
                click: function(){
                    row = $(this).closest('.fixedPublicModuleRow'),
                        id = row.attr('id');
                    if(row.attr('data-close')){
                        var chId = parseInt(wsf.pDom.bodyDom.data("chid"));
                        if (!chId) {
                            return false;
                        }
                        var positionRChIdList = {},
                            positionRChIds = readCookie(user_name+'_positionRChIds');
                        if (positionRChIds) {
                            positionRChIdList = JSON.parse(positionRChIds);
                        }
                        positionRChIdList[chId] = new Date().getTime();
                        writeCookie(user_name+'_positionRChIds', JSON.stringify(positionRChIdList), 1000*24);
                    }
                    row.remove();
                    $('#fixedPublicModuleRowEmpty').remove();
                    that.fixedDom = null;
                }
            },'.fixedPublicModuleRow .rowCloseBtn');
        }
    },
    showFixedDom: function(s, h){
        var that = this;
        if(that.fixedDom && that.fixedDom.length){
            that.fixedDom.each(function(i, dom){
                dom = $(dom);
                dom.attr('data-fixedtop') * 1 <= s ? dom.show() : dom.hide();
            });
        }
    }
};
//跑马灯模块方法
wsf.moveModule = function (obj) {
    obj ? (obj.attr('data-move') * 1 == 1 && obj.moveModule({
        "axis": obj.attr('data-axis'),
        "speed": obj.attr('data-speed'),
        "type": obj.attr('data-type')
    })) : $('div.pic-text-list-module,div.text-list-module').each(function (i, dom) {
        dom = $(dom);
        dom.is(':visible') && dom.attr('data-move') * 1 == 1 && dom.moveModule({
            "axis": dom.attr('data-axis'),
            "speed": dom.attr('data-speed'),
            "type": dom.attr('data-type')
        });
    });
};
/*
fixedShopcar
主要用来显示前台的浮动购物车，和里边的一些方法
 */
wsf.fixedShopcar = function () {
    var fixedShopcar = $("#fixedShopcar"),
        shopcarAlert = fixedShopcar.find("div.shopcar-alert"),
        listBody = fixedShopcar.find("dd.shopcar-list-body"),
        listUl = listBody.find(".shopcar-list-ul");
    listBody.cScroll({
        w: 10,
        tbB: false
    });
    fixedShopcar.on({
        click: function () {
            var that = $(this);
            if (that.data("alertShow") == 1) {
                that.removeData("alertShow");
                shopcarAlert.css({
                    'visibility': 'hidden'
                });
            } else {
                that.data("alertShow", 1);
                shopcarAlert.css({
                    'visibility': 'visible'
                });
            }
        }
    }, ".shopcar-icon");
    fixedShopcar.on({
        click: function () {
            fixedShopcar.find(".shopcar-icon").removeData("alertShow");
            shopcarAlert.css({
                'visibility': 'hidden'
            });
        }
    }, ".shopcar-alert-close");
    listUl.on({
        click: function () {
            var par = $(this).parent().parent();
            par.animate({
                height: 0,
                opacity: 0
            }, 500, function () {
                $(this).remove();
                if (listUl.height() > listBody.height()) {
                    listBody.cScroll({
                        w: 10,
                        tbB: false
                    });
                } else {
                    listBody.cScroll({
                        w: 10,
                        tbB: false
                    });
                    listBody.find(".c-scrollbar").hide();
                }
            });
        }
    }, "span.span-option a");
};
/*
	focus-pic-module
	createDate:2015/04/02
	模块内焦点图函数，主要是模块中的焦点图切换方法
*/
wsf.focusPicModule = function (id) {
    var objArray = id ? $('#' + id).find('div.focus-pic-module') : $("div.focus-pic-module");
    objArray.each(function () {
        var obj = $(this);
        var h = obj.parent().height(),
            w = obj.parent().width(),
            dd = obj.find("dd"),
            dl = obj.find("dl"),
            type = obj.data("type"),
            dir = obj.data("dir"),
            sum = dd.length,
            curi = 0,
            ni = curi + 1,
            times, move, fun;
        dl.css({
            "width": w + "px",
            "height": h + "px",
            "overflow": "hidden"
        });
        dd.css({
            "width": w + "px",
            "height": h + "px"
        });
        //判断设置的方向
        (function () {
            var scss;
            switch (dir) {
                case 1: //left
                    scss = {
                        left: (w + 10) + "px",
                        top: 0
                    };
                    break;
                case 2: //right
                    scss = {
                        left: -(w + 10) + "px",
                        top: 0
                    };
                    break;
                case 3: //top
                    scss = {
                        top: (h + 10) + "px",
                        left: 0
                    };
                    break;
                case 4: //bottom
                    scss = {
                        top: -(h + 10) + "px",
                        left: 0
                    };
                    break;
            }
            dd.each(function () {
                var that = $(this);
                if (that.index() != curi) {
                    that.css(scss);
                } else {
                    that.css({
                        left: 0,
                        top: 0
                    });
                }
            });
        })();
        //判断设置的类型
        if (sum > 1) {
            (function () {
                switch (type) {
                    case 1:
                    case 2:
                        var controls = $('<div class="focus-controls-list"></div>');
                        for (var s = 0; s < sum; s++) {
                            var span;
                            if (type == 1) {
                                span = $('<span>' + (s + 1) + '</span>');
                                if (s === curi) {
                                    span = $('<span class="cur">' + (s + 1) + '</span>');
                                }
                            } else if (type == 2) {
                                span = $('<span></span>');
                                if (s === curi) {
                                    span = $('<span class="cur"></span>');
                                }
                            }
                            controls.append(span);
                        }
                        obj.append(controls);
                        fun = function (i) {
                            obj.find(".focus-controls-list span:eq(" + i + ")").addClass("cur").siblings().removeClass("cur");
                        };
                        obj.on({
                            click: function () {
                                var indexs = $(this).index();
                                if (indexs != curi) {
                                    if (!obj.data("move")) {
                                        move(indexs);
                                    }
                                }
                            }
                        }, ".focus-controls-list span");
                        break;
                    case 3:
                        var nextBtn = $('<span class="focus-pic-next"></span>'),
                            prevBtn = $('<span class="focus-pic-prev"></span>');
                        obj.append(nextBtn, prevBtn);
                        obj.on({
                            mouseenter: function () {
                                var this_ = $(this);
                                if (!this_.data("type3btn")) {
                                    obj.find(".focus-pic-next,.focus-pic-prev").css({
                                        visibility: 'visible'
                                    });
                                    this_.data("type3btn", 1);
                                }
                            },
                            mouseleave: function () {
                                var this_ = $(this);
                                if (this_.data("type3btn") == 1) {
                                    obj.find(".focus-pic-next,.focus-pic-prev").css({
                                        visibility: 'hidden'
                                    });
                                    this_.removeData("type3btn");
                                }
                            }
                        });
                        fun = function (i) {
                            obj.find(".focus-pic-next").data("num", i + 1);
                            obj.find(".focus-pic-prev").data("num", i - 1);
                        };
                        fun(curi);
                        obj.find(".focus-pic-next").on("click", function () {
                            if (!obj.data("move")) {
                                move($(this).data('num'), 1);
                            }
                        });
                        obj.find(".focus-pic-prev").on('click', function () {
                            if (!obj.data("move")) {
                                move($(this).data('num'), 2);
                            }
                        });
                        break;
                }
            })();
            //移动函数
            move = function (c, dir_) {
                var oldDir = dir;
                dir = dir_ || dir;
                if (c == sum) {
                    c = 0;
                }
                if (c < 0) {
                    c = sum - 1;
                }
                obj.data("move", 1);
                var chNum = function () {
                    curi = c;
                    ni = c + 1;
                    obj.data("move", 0);
                };
                if (dir == 1) {
                    $(dd[c]).animate({
                        left: 0
                    }, 1000, function () {
                        chNum();
                    });
                    $(dd[curi]).animate({
                        left: -w
                    }, 1000, function () {
                        $(this).css({
                            left: (w + 10) + "px"
                        });
                        dir = oldDir;
                    });
                }
                if (dir == 2) {
                    $(dd[c]).animate({
                        left: 0
                    }, 1000, function () {
                        chNum();
                    });
                    $(dd[curi]).animate({
                        left: w
                    }, 1000, function () {
                        $(this).css({
                            left: -(w + 10) + "px"
                        });
                        dir = oldDir;
                    });
                }
                if (dir == 3) {
                    $(dd[c]).animate({
                        top: 0
                    }, 1000, function () {
                        chNum();
                    });
                    $(dd[curi]).animate({
                        top: -(h + 10)
                    }, 1000, function () {
                        $(this).css({
                            top: (h + 10) + "px"
                        });
                    });
                }
                if (dir == 4) {
                    $(dd[c]).animate({
                        top: 0
                    }, 1000, function () {
                        chNum();
                    });
                    $(dd[curi]).animate({
                        top: h + 10
                    }, 1000, function () {
                        $(this).css({
                            top: -(h + 10) + "px"
                        });
                    });
                }
                fun(c);
            };
            var setTime = function () {
                times = setInterval(function () {
                    move(ni);
                }, 5000);
            };
            obj.on({
                mouseenter: function () {
                    clearInterval(times);
                },
                mouseleave: function () {
                    setTime();
                }
            });
            setTime();
        }
    });
};
/*
	pic-text-list-module
	createDate:2015/07/20
	changeDate:2016/12/20==>为了解决编辑状态下dom更新后不能用的问题。
	文本列表，图文信息列表模块的方法
*/
wsf.textListModule = function (obj) {
    var fun = function (t) {
        if (!t.data('bindevent')) {
            t.on({
                mouseenter: function () {
                    $(this).addClass("liHover");
                },
                mouseleave: function () {
                    $(this).removeClass("liHover");
                }
            }, 'li');
            (t.hasClass('text-list-module') || t.hasClass('pic-text-list-module')) && (function(){
                var page = t.siblings('.page:first'),
                    mId = t.closest('div.Mo').attr('id').substr(3);
                if(page.length && page.data('page')){
                    var data_ = page.data();
                    page.pagination({
                        skin: data_.skin, //皮肤
                        mId: mId,
                        dataTotal: data_.total,//数据总数
                        pageDataNum: data_.pagedatanum,//每页数据数
                        curPage: data_.curpage,//当前页
                        pageGroups: data_.pagegroups,//显示页码的个数
                        callBack: function(config){
                            $.ajax({
                                'url': '/ajax_tj.php?username='+user_name+'&module_id='+mId+'&page='+config.curPage+'&is_page=1',
                                type: "POST",
                                async: false,
                                cache: false,
                                dataType: 'json',
                                success: function (data) {
                                    if(data.success){
                                        if(t.hasClass('pic-text-list-module')){
                                            t.find('ul').html(data.html);
                                            wsf.interactFun_(t);
                                            if(window.view_pic_lightgallery){
                                                window.view_pic_lightgallery(t);
                                            }
                                        }else if(t.hasClass('text-list-module')){
                                            t.find('ol').html(data.html);
                                        }
                                        wsf.pDom.htmlDom.animate({scrollTop: Math.min(wsf.pDom.htmlDom.scrollTop(), t.closest('.customModuleRow').offset().top)});
                                        wsf.pDom.bodyDom.animate({scrollTop: Math.min(wsf.pDom.bodyDom.scrollTop(), t.closest('.customModuleRow').offset().top)});
                                        t.find('.lazy-loading').each(function(i, dom){
                                            dom = $(dom);
                                            var oImage = new Image();
                                            oImage.onload = function () {
                                                dom.attr('src', this.src).addClass('lazy-loading-animate');
                                                setTimeout(function(){
                                                    dom.removeClass('lazy-loading lazy-loading-animate');
                                                }, 1000);
                                            };
                                            oImage.src = dom.attr('data-original-src');
                                        });
                                    }
                                    if(data.docIds){
                                        GData.docSysIds = data.docIds;
                                        GData.SMAlbumSysIds = GData.SMAlbumIds='';
                                        wsf.albumsDataLoad();
                                    }
                                }
                            });
                        }
                    });
                }
            }());
            t.hasClass('ev-albums-module') && t.on({
                mouseenter: function () {
                    $(this).addClass("hover");
                },
                mouseleave: function () {
                    $(this).removeClass("hover");
                }
            }, '.image-inner');
            t.data('bindevent', 1);
        }
    };
    obj ? fun(obj) : $("div.text-list-module,div.pic-text-list-module,div.ev-albums-module").each(function () {
        fun($(this));
    });
};
/*列表页25的js效果*/
wsf.picTextList_changePic = function () {
    /* 更换图片效果 */
    $('div.b-listpage-pic-text-list-3').on({
        click: function () {
            var t = $(this),
                src = t.find('img').attr('src'),
                lip = t.parents('li'),
                picimg = lip.find(".pics img");
            picimg.attr('src', src);
            t.addClass('pl-list-dd-cur').siblings().removeClass('pl-list-dd-cur');
        }
    }, '.pl-small-pic-list .pl-list-dd');
    /*显示隐藏收藏*/
    $('div.b-listpage-pic-text-list-3').on({
        mouseenter: function () {
            var t = $(this),
                plCollect = t.find(".pl-collect");
            plCollect.animate({
                right: 0
            }, 100);
        },
        mouseleave: function () {
            var t = $(this),
                plCollect = t.find(".pl-collect");
            plCollect.animate({
                right: -100
            }, 100);
        }
    }, '.pics');

    $('div.b-listpage-pic-text-list-3, .collect_p').on({
        click: function (ev) {
            var t = $(this),
                div = t.children('div');

            var timestamp = Date.parse(new Date());
            var url = '/dom/user_collect_add.php?timestamp=' + timestamp;
            var is_detail = t.data("detail");
            var data = {
                'title': t.data("name"),
                'type': t.data("type"),
                'doc_id': t.data("id"),
                'channel_id': t.data("chid"),
                'username': user_name,
                'wap': 0
            };
            $.ajax({
                'url': url,
                type: "POST",
                async: false,
                cache: false,
                data: data,
                success: function (data) {
                    if (data == 1) {
                        //`div.attr('class','no-collect').find("b").text('收藏');
                        div.attr('class', 'yes-collect').find("b").text($weisiteLa.YiShouCang);
                        wsf.f.alertWindow($weisiteLa.ShouCangChengGongGeRenZhongXinKan, 2);
                    } else if (data == 2) {
                        wsf.f.alertWindow($weisiteLa.ShouCangShiBaiMeiDengLuChongXinShouCang, 0);
                        window.location.href = "/dom/denglu.php?username=" + user_name;
                        return false;
                    } else if (data == 3) {
                        if (is_detail) {
                            div.attr('class', 'no-collect').find("b").text($weisiteLa.ShouCang);
                            wsf.f.alertWindow($weisiteLa.QuXiaoShouCangChengGong, 0);
                        } else {
                            div.attr('class', 'yes-collect').find("b").text($weisiteLa.YiShouCang);
                            wsf.f.alertWindow($weisiteLa.ShouCangGuoGeRenZhongXinKan, 0);
                        }
                        return false;
                    } else if (data == 4) {
                        wsf.f.alertWindow($weisiteLa.CanShuCuoWu, 0);
                        return false;
                    }
                }
            });
            return false;
        }
    }, '.pl-collect');
};
/* 交互动画的json对象 */
wsf.interactJ = {
    a_1: {
        typeJ: {
            t_1: 'int-onlyimg-larger',
            t_2: 'int-onlyimg-small',
            t_3: 'int-onlyimg-move-left',
            t_4: 'int-onlyimg-move-right',
            t_5: 'int-onlyimg-move-top',
            t_6: 'int-onlyimg-move-down',
            t_7: 'int-onlyimg-rotate-left',
            t_8: 'int-onlyimg-rotate-right'
        },
        aClass: 'int-dom',
        addDom: function (obj, j) {
            obj.addClass(this.aClass + ' ' + this.typeJ['t_' + j.t]);
        }
    },
    a_2: {
        typeJ: {
            t_1: 'int-bigglass-fade',
            t_2: 'int-bigglass-magnify',
            t_3: 'int-bigglass-shrink',
            t_4: 'int-bigglass-slide-left',
            t_5: 'int-bigglass-slide-right',
            t_6: 'int-bigglass-slide-up',
            t_7: 'int-bigglass-slide-down'
        },
        aClass: 'int-dom',
        addDom: function (obj, j) {
            obj.addClass(this.aClass + ' ' + this.typeJ['t_' + j.t]);
            if (this.typeJ['t_' + j.t]) {
                var domhtml = '<div class="int-add-dom"><div class="bgzz"></div><div class="figcaption"><img src="/images/VNew/magnifying_glass_img.png" /></div></div>';
                obj.append(domhtml);
            }
        }
    },
    a_3: {
        typeJ: {
            t_1: 'int-changeimg-fade',
            t_2: 'int-changeimg-magnify',
            t_3: 'int-changeimg-shrink',
            t_4: 'int-changeimg-slide-left',
            t_5: 'int-changeimg-slide-right',
            t_6: 'int-changeimg-slide-up',
            t_7: 'int-changeimg-slide-down',
            t_8: 'int-changeimg-slide-leftup',
            t_9: 'int-changeimg-slide-rightup',
            t_10: 'int-changeimg-slide-leftdown',
            t_11: 'int-changeimg-slide-rightdown',
            t_12: 'int-changeimg-push-left',
            t_13: 'int-changeimg-push-right',
            t_14: 'int-changeimg-push-up',
            t_15: 'int-changeimg-push-down',
            t_16: 'int-changeimg-hinge-left',
            t_17: 'int-changeimg-hinge-right',
            t_18: 'int-changeimg-hinge-up',
            t_19: 'int-changeimg-hinge-down',
            t_20: 'int-changeimg-flip-horiz',
            t_21: 'int-changeimg-flip-vert',
            t_22: 'int-changeimg-flip-diag-l',
            t_23: 'int-changeimg-flip-diag-r',
            t_24: 'int-changeimg-shutter-out-horiz',
            t_25: 'int-changeimg-shutter-out-vert',
            t_26: 'int-changeimg-shutter-out-diag-l',
            t_27: 'int-changeimg-shutter-out-diag-r'
        },
        aClass: 'int-dom',
        addDom: function (obj, j) {
            obj.addClass(this.aClass + ' ' + this.typeJ['t_' + j.t]);
            if (this.typeJ['t_' + j.t]) {
                var nImgUrl = obj.data('nimgurl');
                var domhtml = '<div class="int-add-dom"><div class="bgzz"></div><div class="figcaption"><img src="' + nImgUrl + '" /></div></div>';
                obj.append(domhtml);
            }
        }
    },
    a_4: {
        typeJ: {
            t_1: 'int-showAttr-fade',
            t_2: 'int-showAttr-magnify',
            t_3: 'int-showAttr-shrink',
            t_4: 'int-showAttr-slide-left',
            t_5: 'int-showAttr-slide-right',
            t_6: 'int-showAttr-slide-up',
            t_7: 'int-showAttr-slide-down',
            t_8: 'int-showAttr-slide-leftup',
            t_9: 'int-showAttr-slide-rightup',
            t_10: 'int-showAttr-slide-leftdown',
            t_11: 'int-showAttr-slide-rightdown',
            t_12: 'int-showAttr-hinge-left',
            t_13: 'int-showAttr-hinge-right',
            t_14: 'int-showAttr-hinge-up',
            t_15: 'int-showAttr-hinge-down',
            t_16: 'int-showAttr-shutter-out-horiz',
            t_17: 'int-showAttr-shutter-out-vert',
            t_18: 'int-showAttr-shutter-out-diag-l',
            t_19: 'int-showAttr-shutter-out-diag-r'
        },
        aClass: 'int-dom',
        addDom: function (obj, j) {
            obj.addClass(this.aClass + ' ' + this.typeJ['t_' + j.t]);
            if (this.typeJ['t_' + j.t]) {
                var ntitle = obj.data('ntitle'),
                    nintro = obj.data('nintro'),
                    picH = obj.height();
                ntitle = ntitle ? ntitle : '这是标题';
                nintro = nintro ? nintro : '这是介绍';
                var domhtml = $('<div class="int-add-dom"><div class="bgzz"></div><div class="figcaption"><div class="show-attr"><h3>' + ntitle + '</h3><p>' + nintro + '</p></div></div></div>');
                obj.append(domhtml);
                var showAttr = domhtml.find('.figcaption'),
                    attrH = showAttr.height(),
                    attrT = (picH - attrH) / 2;
                showAttr.css({
                    top: attrT + 'px'
                });
            }
        }
    },
    a_5: {
        typeJ: {
            t_1: 'int-showTitle-fade',
            t_2: 'int-showTitle-magnify',
            t_3: 'int-showTitle-shrink',
            t_4: 'int-showTitle-slide-left',
            t_5: 'int-showTitle-slide-right',
            t_6: 'int-showTitle-slide-up',
            t_7: 'int-showTitle-slide-down',
            t_8: 'int-showTitle-slide-leftup',
            t_9: 'int-showTitle-slide-rightup',
            t_10: 'int-showTitle-slide-leftdown',
            t_11: 'int-showTitle-slide-rightdown',
            t_12: 'int-showTitle-shutter-out-horiz',
            t_13: 'int-showTitle-shutter-out-vert'
        },
        aClass: 'int-dom',
        addDom: function (obj, j) {
            obj.addClass(this.aClass + ' ' + this.typeJ['t_' + j.t]);
            if (this.typeJ['t_' + j.t]) {
                var ntitle = obj.data('ntitle');
                ntitle = ntitle ? ntitle : '这是标题';
                var domhtml = '<div class="int-add-dom"><div class="bgzz"></div><div class="figcaption"><div class="show-attr"><h3>' + ntitle + '</h3></div></div></div>';
                obj.append(domhtml);
            }
        }
    }
};
/* 交互动画的方法 */
wsf.interactFun = function () {
    var picTextListModule = $("div.pic-text-list-module"),
        picTextModule = $("div.pic-text-module"),
        picModule = $('div.pic-module'),
        evPicModule = $('div.ev-pic');
    /* 图文信息，橱窗的图片交互 */
    picTextListModule.each(function () {
        var that = $(this),
            interact = that.data('interact');
        interact = interact ? wsf.f.s_j(interact) : interact;
        if (interact) {
            if (wsf.interactJ['a_' + interact.a]) {
                that.find("li").each(function () {
                    var t = $(this),
                        pic = t.find('.pic');
                    wsf.interactJ['a_' + interact.a].addDom(pic, interact);
                });
            }
        }
    });
    /* 图文混排的图片交互 */
    picTextModule.each(function () {
        var that = $(this),
            interact = that.data('interact');
        interact = interact ? wsf.f.s_j(interact) : interact;
        if (interact) {
            if (wsf.interactJ['a_' + interact.a]) {
                var pic = that.find('.pic');
                wsf.interactJ['a_' + interact.a].addDom(pic, interact);
            }
        }
    });
    /* 单张图片交互 */
    picModule.each(function () {
        var that = $(this),
            interact = that.data('interact');
        interact = interact ? wsf.f.s_j(interact) : interact;
        if (interact) {
            if (wsf.interactJ['a_' + interact.a]) {
                var pic = that.find('.pic');
                wsf.interactJ['a_' + interact.a].addDom(pic, interact);
            }
        }
    });
};
wsf.showOneBigPic = function(picUrl){
    if(!picUrl){
        return;
    }
    var bigPicBox = $('<div class="show-big-pic-box"></div>');
    var bigPic = $('<div class="big-pic-box"><img src="' + picUrl + '" class="show-big-pic" /></div>');
    var bigPicZZ = $('<div class="big-pic-zz"></div>');
    bigPicBox.append(bigPicZZ);
    bigPicBox.append(bigPic);
    $('body').append(bigPicBox);
    bigPicBox.on({
        click: function(){
            $(this).remove();
        }
    });
};
/* 交互动画类型4的属性居中方法 */
wsf.interactFun_ = function (obj) {
    var fun = function (t) {
        t = $(t);
        var figcaption = t.find('.figcaption'),
            top_ = ((t.parent().parent().hasClass('pic-module') ? t.closest('.MoBodyC').height() : t.height()) - figcaption.height()) / 2;
        figcaption.css({
            top: top_ + 'px'
        });
    };
    if(obj && obj.length){
        obj.find('.pic[class*="int-showAttr-"]').each(function (i, d) {
            fun(d);
        });
    }else{
        $('.pic[class*="int-showAttr-"]').each(function (i, d) {
            fun(d);
        });
        $('.ev-pic[class*="int-showAttr-"]').each(function (i, d) {
            fun(d);
        });
    }
};
/*
	catalogList
	树形分类目录，主要用于模块的分类目录js，包括上下移动
 */
wsf.catalogList = function (obj_) {
    var catalogH = function (obj) {
            var oneDl = obj.children("dl.oneClassList");
            if (obj.height() < oneDl.height() || parseInt(oneDl.css("margin-top"), 10) < 0) {
                obj.find("big.but").show();
            } else {
                obj.find("big.but").hide();
            }
        },
        move = function (obj, dir) {
            var oneDl = obj.find("dl.oneClassList"),
                t_h = obj.height(),
                dl_t = Math.abs(parseInt(oneDl.css("margin-top"), 10)),
                dl_sh = oneDl.height() - dl_t,
                judge;

            if (dir == 'up') {
                judge = (dl_sh - t_h) > 0 ? true : false;
            } else if (dir == 'down') {
                judge = dl_t > 0 ? true : false;
                t_h = -t_h;
            }

            if (judge) {
                if (oneDl.data('moving') != 1) {
                    oneDl.data('moving', 1);
                    oneDl.animate({
                        marginTop: -(dl_t + t_h / 3)
                    }, 500, function () {
                        oneDl.removeData('moving', 1);
                    });
                }
            }
        },
        fun = function (t) {
            if (!t.data('bindevent')) {
                if (!t.hasClass('sidebarLists')) {
                    t.css({
                        height: t.parent().height() + "px",
                        width: t.parent().width() + "px"
                    });
                }
                //给类别绑定展开闭合子类事件
                t.on({
                    click: function(ev){
                        var $this = $(this),
                            clicktype = $this.data('clicktype');
                        if(clicktype == 1 || (ev.target && ev.target.tagName.toLocaleLowerCase() === 'code')){
                            var pDt = $this.closest('dt'),
                                num;
                            if (pDt.hasClass('oneClassT')) {
                                num = 'one';
                            } else if (pDt.hasClass('twoClassT')) {
                                num = 'two';
                            } else if (pDt.hasClass('threeClassT')) {
                                num = 'three';
                            }
                            if (pDt.hasClass(num + 'ClassTopen')) {
                                pDt.removeClass(num + 'ClassTopen').next("dd").removeClass(num + 'ClassCopen');
                                pDt.find('code').removeClass("open");
                            } else {
                                pDt.addClass(num + 'ClassTopen').next("dd").addClass(num + 'ClassCopen');
                                pDt.find('code').addClass("open");
                            }
                            if (!t.hasClass('sidebarLists')) {
                                catalogH(t);
                            }
                            return false;
                        }

                    },
                    mouseenter: function () {
                        var pDt = $(this);
                        if (pDt.hasClass('oneClassT')) {
                            num = 'one';
                        } else if (pDt.hasClass('twoClassT')) {
                            num = 'two';
                        } else if (pDt.hasClass('threeClassT')) {
                            num = 'three';
                        }
                        pDt.addClass(num + 'ClassThover');
                    },
                    mouseleave: function () {
                        var pDt = $(this);
                        if (pDt.hasClass('oneClassT')) {
                            num = 'one';
                        } else if (pDt.hasClass('twoClassT')) {
                            num = 'two';
                        } else if (pDt.hasClass('threeClassT')) {
                            num = 'three';
                        }
                        pDt.removeClass(num + 'ClassThover');
                    }
                }, 'dt');

                if (!t.hasClass('sidebarLists')) {
                    // 给上下按钮绑定事件
                    t.on({
                        click: function () {
                            if ($(this).hasClass('upBut')) {
                                move(t, 'down');
                            } else if ($(this).hasClass('downBut')) {
                                move(t, 'up');
                            }
                        }
                    }, "big.but");

                    // 给元素本身绑定事件
                    t.on({
                        mouseenter: function () {
                            var this_ = $(this),
                                dl = this_.find("dl.oneClassList"),
                                dl_h = dl.height(),
                                t_h = this_.height();
                            if (dl_h > t_h || parseInt(dl.css("margin-top"), 10) < 0) {
                                this_.find("big.but").show();
                            }
                        },
                        mouseleave: function () {
                            $(this).find("big.but").hide();
                        }
                    });
                }
                t.data('bindevent', 1);
            }
        };
    if (obj_) {
        fun(obj_);
    } else {
        $("div.catalogList").each(function () {
            fun($(this));
        });
    }
};
/*
	menu-catalog-module
	抽屉型分类目录主要用来展示更多分类
*/
wsf.menuCatalogModule = function (obj_) {
    var fun = function (that) {
        if (!that.data('bindevent')) {
            var menuCatalog = that,
                Mo = menuCatalog.parents("div.Mo"),
                MoId = Mo.attr("id"),
                catalog = $("#menuCatalogMore_" + MoId.substr(3)),
                leveObj, hideTimeFun,
                moreInner = catalog.find(".catalog-more-inner"),
                gap = catalog.find("big.gap"),
                hideCatalog = function () {
                    hideTimeFun = setTimeout(function () {
                        leveObj.removeClass('one-class-hover').data('h', 0);
                        catalog.css("width", 0).data('visible', 0);
                    }, 500);
                };
            catalog.on({
                mouseenter: function () {
                    clearTimeout(hideTimeFun);
                },
                mouseleave: function () {
                    hideCatalog();
                }
            });
            menuCatalog.on({
                mouseenter: function () {

                },
                mouseleave: function () {
                    hideCatalog();
                }
            });
            menuCatalog.on({
                mouseenter: function () {
                    var tObj = $(this);
                    clearTimeout(hideTimeFun);
                    leveObj = tObj.parent();
                    if (!leveObj.data('h')) {
                        leveObj.addClass("one-class-hover").data('h', 1).siblings('dd.one-class-hover').removeClass('one-class-hover').data('h', 0);
                        var indexs = leveObj.index(),
                            that_t = leveObj.position().top,
                            that_h = leveObj.outerHeight(),
                            inner_mt = parseInt(tObj.css("margin-top")),
                            inner_tbw = parseInt(tObj.css("borderTopWidth")),
                            inner_bbw = parseInt(tObj.css("borderBottomWidth")),
                            inner_h = tObj.innerHeight(),
                            l = menuCatalog.offset().left,
                            t = menuCatalog.offset().top,
                            gap_h = inner_h,
                            gap_t = that_t + inner_tbw + inner_mt,
                            moreList = catalog.find(".catalog-more-list:eq(" + indexs + ")");
                        l = l + menuCatalog.width() - 2;
                        gap.css({
                            height: gap_h + "px",
                            top: gap_t + "px"
                        });
                        moreList.siblings().hide();
                        if (moreList.find("dl").length >= 1) {
                            moreList.show();
                            catalog.css({
                                visibility: 'visible'
                            });
                        } else {
                            catalog.css({
                                visibility: 'hidden'
                            });
                        }
                        moreInner.css({
                            minHeight: gap_h + "px",
                            marginTop: that_t + inner_mt + 'px'
                        });
                        catalog.css({
                            left: l + "px",
                            top: t + "px"
                        });
                        if (!catalog.data('visible')) {
                            catalog.animate({
                                width: moreInner.outerWidth()
                            }, 200, function () {
                                catalog.css({
                                    width: "auto"
                                });
                            });
                            catalog.data('visible', 1);
                        }
                    }
                },
                mouseleave: function () {
                    //$(this).removeClass("one-class-hover");
                }
            }, ".one-class-inner");
            menuCatalog.data('bindevent', 1);
        }
    };
    if (obj_) {
        fun(obj_);
    } else {
        $("div.menu-catalog-module").each(function () {
            fun($(this));
        });
    }
};
/*
	tableModule
	createDate:2015/05/11
	用来给表格添加class
*/
wsf.tableModule = function () {
    var tableModule = $('div.tableModule');
    tableModule.each(function () {
        $(this).find('tr:first').addClass('trHead');
    });
};
/*
	tab-switch-module
	createDate:2015/03/24
	模块标签项切换功能，现在主要用在最终页的标签切换
*/
wsf.tabSwitchModule = function () {
    var tabSwitch = $("div.tab-switch-module");
    if(tabSwitch.length){
        tabSwitch.each(function(i, dom){
            dom = $(dom);
            var tabT = dom.find("div.tab-switch-t").filter(':first'),
            tabCItem = dom.find("div.tab-switch-c .tab-c-item"),
            tabType = dom.data('type');
            tabT.on({
                click: function () {
                    if (tabT.data('noswitch') != 1) {
                        var that = $(this),
                            indexs = that.index();
                        that.find("span").addClass("active");
                        that.siblings("li").find("span").removeClass("active");
                        tabCItem.each(function (i, itemDom) {
                            itemDom = $(itemDom);
                            i === indexs ? itemDom.addClass("tab-c-item-active") : itemDom.removeClass("tab-c-item-active");
                            if(tabType == 'registerPage' && i === indexs){
                                if(itemDom.find('.vcode-img').length){
                                    var v_id = itemDom.find('.vcode-img').data('id');
                                    !is_action && getVcode2(v_id);
                                }
                            }
                        });
                    }
                }
            }, "li");
        });
    }
};
/*
	form-module
	createDate:2017/09/20
	表单模块方法
*/
wsf.formModule = {
    defaultValVerify: function (formModule) {
        formModule = formModule || $('div.form-module');
        formModule.each(function (i, dom) {
            dom = $(dom);
            dom.on('blur focus', 'input[type="text"],textarea', function (event) {
                if(!('placeholder' in document.createElement('input'))){
                    var $this = $(this),
                        defaultVal = $this.attr('data-defaultval'),
                        changeVal = $this.val();
                    switch (event.type) {
                        case 'focusout':
                            (!changeVal || changeVal == defaultVal) && $this.val(defaultVal);
                            break;
                        case 'focusin':
                            (changeVal == defaultVal) && $this.val('');
                            break;
                    }
                }
            });
            dom.on({
                click:function(){

                }
            },'input[type="submit"]');
        });
    },
    submit_verify: function (formModule, j) {
        formModule = formModule || $('div.form-module from');
        formModule.find('input[type="button"]').removeAttr('onclick');
        formModule.validationEngine('attach', {
            promptPosition: 'bottomLeft'
        });
        var formId = formModule.attr('id'),
            moId = formModule.closest('.Mo').attr('id'),
            message = null;
        if(j){
            $.each(j, function(i, v){
                if(v['MId'] == formId.substring(10)){
                    message = v;
                }
            });
        }
        formModule.on({
            click: function(){
                if(message.msg){
                    alert(decodeHtmlEntity(message.msg));
                }else{
                    var $this = $(this);
                    if(formModule.validationEngine("validate")){
                        // 判断是否显示图案验证 0表示显示，1表示不显示
                        if(!(message.showVerification*1)){
                            if(!tncode_div){
                                tncode.init('MessageBut',1);
                                tncode_div =true;
                            }else{
                                tncode.show();
                            }
                            $TN.onsuccess(function(){
                                formModule.append('<input type="hidden" name="tn_r" value="'+tncode._mark_offset+'">');
                                formModule.find('input[name="isSubmit"]').val(1);
                                formModule.submit();
                                $this.closest('.form-button').html('<font style="color:red;font-size:14px;">' + message.SubmitMsg + '...</font>');
                            });
                        }else{
                            formModule.find('input[name="isSubmit"]').val(1);
                            formModule.submit();
                        }
                    }
                }
            }
        }, 'input[type="button"]');
    }
};
// 自由编辑模块
wsf.customEditModule = function () {
    var customEditModule = $('div.custom-edit-module').each(function () {
        var t = $(this),
            tp = t.parent();
        t.css({
            height: tp.height()
        });
    });
};
/*
	video-module
	createDate:2017/11/01
	视频模块的方法
*/
wsf.videoModule = {
    chplayerArr: [],
    load: function (obj) {
        var that = this;
        var fun = function (dom) {
            var chplayerObj = new chplayer({
                container: '#' + dom.attr('id'),
                variable: 'player',
                autoplay: !!(dom.attr('data-autoplay')*1),
                volume: !!(dom.attr('data-autoplay')*1) ? 0 : 1,
                poster: dom.attr('data-pic') || '', //封面图片地址
                video: dom.attr('data-url'),
                loop: !!(dom.attr('data-loopplay')*1)
            });
            that.chplayerArr.push(chplayerObj);
            try {
                chplayerObj.addListener('play', function(){
                    that.chplayerArr.forEach(function(v){
                        if(v !== chplayerObj){
                            if(!v.getMetaDate()['paused']){
                                v.pause();
                            }
                        }
                    })
                });
            } catch (error) {
                console.log(error);
            }
        };
        obj ? fun(obj) : $("div.video-module,div.ev-custom-video").each(function (i, dom) {
            dom = $(dom);
            if(dom.css('visibility') !== 'hidden'){
                fun(dom);
            }
        });
    }
};

/**
 * baidu map init
 *
 * @param id
 * @param x
 * @param y
 * @param con
 * @param level
 */
wsf.baiduInit = function (id, x, y, con, level) {
    con = $.trim(con);
    if (user_name == 'samui') {
        x = 99.997234;
        y = 9.501727;
    }
    if (parseInt(level) < 1 || isNaN(parseInt(level))) {
        level = 15;
    }

    var map = new BMap.Map("container_" + id);
    var point  = new BMap.Point(x, y);
    var marker = new BMap.Marker(point);
    var navCtrl = new BMap.NavigationControl({
        type: BMAP_NAVIGATION_CONTROL_ZOOM
    });

    map.centerAndZoom(point, level);
    map.disableDoubleClickZoom();
    map.addControl(navCtrl);
    map.addOverlay(marker);

    if (con) {
        con = "<p>" + con + "</p>";
    }

    //创建检索信息窗口对象
    var searchInfoWindow = new BMapLib.SearchInfoWindow(map, con, {
        title  : '',       //标题
        width  : 290,             //宽度
        panel  : "panel",         //检索结果面板
        enableAutoPan : true,     //自动平移
        enableSendToPhone: false, //是否显示发送到手机按钮
        searchTypes   :[
            // BMAPLIB_TAB_SEARCH,   //周边检索
            // BMAPLIB_TAB_TO_HERE,  //到这里去
            // BMAPLIB_TAB_FROM_HERE //从这里出发
        ]
    });

    if (wsf.baiduAlter === false) {
        wsf.baiduAlter = true;
        searchInfoWindow.open(marker);
    }

    marker.addEventListener('click', function () {
        searchInfoWindow.open(marker);
    });
};

/**
 * chart init
 *
 * @param type
 * @param id
 * @param start
 * @param end
 * @param step
 */
wsf.chartInit = function (type, id, start, end, step, data) {
    var chartDiv = $('#chartDiv' + id);
    if(chartDiv.is(':visible') && !chartDiv.data('loaddata')){
        chartDiv.html('');
        var chart = {};
        type = parseInt(type);
        if (type === 3) {
            chart = new dhtmlXChart({
                view: "line",
                container: "chartDiv" + id,
                value: "#value#",
                item: {
                    borderColor: "#1293f8",
                    color: "#ffffff"
                },
                line: {
                    color: "#1293f8",
                    width: 3
                },
                xAxis: {
                    template: "#title#"
                },
                offset: 0,
                yAxis: {
                    start: start,
                    end: end,
                    step: step,
                    template: function (obj) {
                        return obj;
                    }
                }
            });
        } else if (type === 2) {
            chart = new dhtmlXChart({
                view: "pie",
                container: "chartDiv" + id,
                value: "#value#",
                label: "#title#",
                pieInnerText: "#value#",
                shadow: 0,
                legend: {
                    width: 75,
                    align: "right",
                    valign: "middle",
                    template: "#title#"
                }
            });
        } else {
            chart = new dhtmlXChart({
                view: "bar",
                container: "chartDiv" + id,
                value: "#value#",
                radius: 0,
                border: true,
                xAxis: {
                    template: "#title#"
                },
                yAxis: {
                    start: start,
                    end: end,
                    step: step,
                    template: function (obj) {
                        return obj;
                    }
                }
            });
        }
        chart.parse(data, "json");
        chartDiv.data('loaddata',1);
    }
};

/*
	row-classify-module
	createDate:2015/08/16
	横向产品分类列表模块的方法
*/
wsf.classifyModule = function () {
    var classifyModule = $("div.classify-module");
    if(classifyModule.length){
        var classList = ['classify-hover','pic-classify-hover'];
        classifyModule.on({
            'mouseenter mouseleave': function (ev) {
                var $this = $(this),curClass = '';
                switch(true){
                    case $this.hasClass('big-classify'):
                    case $this.hasClass('small-classify'):
                        curClass = classList[0];
                        break;
                    case $this.hasClass('small-pic-classify'):
                        curClass = classList[1];
                        break;
                }
                ev.type === 'mouseenter' ? $this.addClass(curClass) : $this.removeClass(curClass);
            }
        }, ".big-classify,.small-classify,.small-pic-classify");
        classifyModule.on('click', '.open-close-btn', function(){
            var $this = $(this),
                action = $this.data('action');
            if(action === 'open'){
                $this.closest('.small-classify-wrap').addClass('open-state');
                $this.data('action', 'close');
            }else{
                $this.closest('.small-classify-wrap').removeClass('open-state');
                $this.data('action', 'open');
            }

        });
    }
};
/*选择列表*/
wsf.selectState = function () {
    $("#filterSort").on({
        mouseenter: function () {
            $(this).addClass("select-state-hover");
        },
        mouseleave: function () {
            $(this).removeClass('select-state-hover');
        }
    }, ".select-state");
};
/*
    manual-move-module
    createDate:2017/08/16
    手动移动模块
*/
wsf.manualMoveModule = function (obj) {
    var manual;
    manual = obj ? obj : $("div.manual-move-module");
    manual.each(function () {
        var t = $(this),
            li = t.find('li'),
            liNum = li.length,
            liW = li.outerWidth(),
            ulP = t.find(".pic-text-list-module");
        li.width(liW);
        ulP.width(liW * liNum).height(li.outerHeight());
    });
    manual.on({
        click: function () {
            var t = $(this),
                id = t.data('id'),
                obj = t.parent();
            moveFun(id, obj);
        }
    }, 'em.prev-move,em.next-move');
    manual.on({
        mouseenter: function () {
            var t = $(this);
            if(t.width() < t.find(".pic-text-list-module").width()){
                t.find('em.prev-move,em.next-move').css('visibility', 'visible');
            }
        },
        mouseleave: function () {
            var t = $(this);
            t.find('em.prev-move,em.next-move').css('visibility', 'hidden');
        }
    });
    var moveFun = function (dir, obj) {
        var curManual = obj,
            list = curManual.find('.manual-move-body'),
            ulP = curManual.find(".pic-text-list-module"),
            li = ulP.find('li'),
            liNum = li.length,
            liWidth = li.outerWidth(),
            showNum = Math.floor((list.width()) / liWidth),
            ulPW = ulP.width(),
            ulPLeft = Math.abs(parseInt(ulP.css('margin-left'))),
            move = showNum * liWidth,
            newMove = ulPW - (ulPLeft + move),
            nowMove;
        if (curManual.data('move') == 1) return false;
        curManual.data('move', 1);
        if (dir == 'next') {
            if (newMove > 0 && newMove > move) {
                nowMove = move;
            } else {
                nowMove = newMove;
            }
            if (ulPW > move) {
                ulP.animate({
                    "marginLeft": "-=" + nowMove
                }, 500, function () {
                    curManual.removeData('move');
                });
            }
        } else {
            nowMove = ulPLeft > move ? move : ulPLeft;
            ulP.animate({
                "marginLeft": "+=" + nowMove
            }, 500, function () {
                curManual.removeData('move');
            });
        }
    };
};
/*
    absolute-menu-catalog
    createDate:2016/08/16
    商城分类模块
*/
wsf.absoluteModuleWrapBind = function () {
    var absMoInner = $("#absolute_module_inner");
    absMoInner.on({
        mouseenter: function () {
            var t = $(this);
            if (t.hasClass('absolute-menu-catalog')) {
                wsf.absoluteMenuCatalog();
            }
        }
    }, '.absolute-module');
};
wsf.absoluteMenuCatalog = function () {
    var menuCatalog = $('div.absolute-menu-catalog');
    if (menuCatalog.length > 0) {
        menuCatalog.each(function () {
            var t = $(this);
            if (!t.data('bindevent')) {
                var open = t.data('open'),
                    amc = t.find('.a-m-c'),
                    mch = t.find('.m-c-h'),
                    mcb = t.find('.m-c-b'),
                    mcm = t.find('.m-c-m'),
                    gap = mcm.find('.gap');
                t.on({
                    mouseleave: function () {
                        if (!open) {
                            amc.removeClass('a-m-c-open');
                            mch.removeClass('m-c-h-open');
                        }
                        if (mcm.data('open')) {
                            mcm.data('open', 0).css({
                                display: 'none'
                            });
                        }
                        mcb.find('.o-l-e').each(function () {
                            var ot = $(this);
                            if (ot.data('open')) {
                                ot.removeClass('o-l-e-open').data('open', 0);
                                var tNimg = ot.find('.o-l-h-n img');
                                tNimg.attr('src', tNimg.data('src'));
                            }
                        });
                    }
                });
                t.on({
                    mouseenter: function () {
                        if (!open) {
                            amc.addClass('a-m-c-open');
                            mch.addClass('m-c-h-open');
                        }
                    }
                }, '.m-c-h-i');
                t.on({
                    mouseenter: function () {
                        var that = $(this),
                            o_openId = that.data('id'),
                            tagOpen = 0;
                        if (!that.data('open')) {
                            /*更换小图标地址*/
                            that.addClass('o-l-e-open').data('open', 1);
                            var tNimg = that.find('.o-l-h-n img'),
                                thatSib = that.siblings('.o-l-e-open'),
                                sibNimg = thatSib.find('.o-l-h-n img');
                            tNimg.attr('src', tNimg.data('hsrc'));
                            thatSib.removeClass('o-l-e-open').data('open', 0);
                            sibNimg.attr('src', sibNimg.data('src'));
                            var oleiBTW = parseInt(that.find('.o-l-e-i').css('borderTopWidth')),
                                tHeight = that.find('.o-l-e-i').height(),
                                mcmiBTW = parseInt(mcm.find('.m-c-m-i').css('borderTopWidth')),
                                tTop = that.position().top - mch.height();
                            oleiBTW = (isNaN(oleiBTW) ? 0 : oleiBTW);
                            mcmiBTW = (isNaN(mcmiBTW) ? 0 : mcmiBTW);
                            gap.css({
                                height: tHeight + 'px',
                                top: (tTop + oleiBTW) + 'px'
                            });
                            if (t.data('mcmfixed')) {
                                mcm.find('.m-c-m-i').css({
                                    'margin-top': (tTop - mcmiBTW) + 'px'
                                });
                            }
                            mcm.find('.m-c-m-e').each(function () {
                                var iThis = $(this),
                                    m_openId = iThis.data('id');
                                if (m_openId == o_openId) {
                                    iThis.addClass('m-c-m-e-open');
                                    tagOpen = 1;
                                } else {
                                    iThis.removeClass('m-c-m-e-open');
                                }
                            });
                            if (tagOpen == 1) {
                                if (mcm.data('open') != 1) {
                                    mcm.css({
                                        display: 'block',
                                        visibility: 'hidden'
                                    });
                                    var macT = mch.height(),
                                        mcmW = mcm.width(),
                                        mcbiBRW = parseInt(mcb.find('.m-c-b-i').css('border-right-width')),
                                        mcmiBRW = parseInt(mcm.find('.m-c-m-i').css('border-left-width')),
                                        mcmL = mcb.width() - (isNaN(mcbiBRW) ? 0 : mcbiBRW) - (isNaN(mcmiBRW) ? 0 : mcmiBRW);
                                    mcm.css({
                                        visibility: 'visible',
                                        width: 0,
                                        left: mcmL + 'px',
                                        top: macT + 'px'
                                    });
                                    mcm.animate({
                                        width: mcmW
                                    }, 200).data('open', 1);
                                }
                            } else {
                                mcm.css({
                                    display: 'none'
                                }).data('open', 0);
                            }
                        }
                    }
                }, '.o-l-e');
                t.data('bindevent', 1);
            }
        });
    }
};
/*
	ev-tab-container
	createDate:2017/09/24
	新模块标签项切换功能
*/
wsf.evTabContent = {
    swtichFun: function (targetDom, targetData) {
        var that = this,
            CModulePA = targetDom.closest('.CModulePA'),
            evModuleEditBox = targetDom.find('.ev-module-edit-box:first');
        if (!evModuleEditBox.hasClass('ev-tab-active')) {
            var groupmark = targetDom.attr('data-groupmark'),
                tabmark = targetDom.attr('data-tabmark');
            evModuleEditBox.addClass('ev-tab-active');
            var oneClass = '.ev-module-edit';
            var twoClass = '.ev-tab-container-two';
            switch(targetData.big*1){
                case 9:
                    oneClass = '.ev-module-edit.ev-tab-container-one';
                    twoClass = '.ev-module-edit.ev-tab-container-two';
                break;
                case 26:
                    oneClass = '.ev-module-edit.ev-interact-container-one';
                    twoClass = '.ev-module-edit.ev-interact-container-two';
                    console.log(targetData.triggerShow, 'targetData.triggerShow');
                    if(targetData.small == 2){
                        if(targetData.triggerShow == 0){
                            targetDom.addClass('edit-hidden').find('.ev-module-edit-box:first').attr('data-loadanimate', 0);
                        }
                    }
                break;
            }
            CModulePA.children(oneClass).not(targetDom).each(function (i, curDom) {
                curDom = $(curDom);
                if(curDom.attr('data-groupmark') === groupmark){
                    curDom.find('.ev-module-edit-box:first').removeClass('ev-tab-active');
                }
            });
            CModulePA.children(twoClass).each(function (i, curDom) {
                curDom = $(curDom);
                if (curDom.attr('data-groupmark') === groupmark) {
                    if(curDom.hasClass('ev-tab-active')){
                        curDom.addClass('none').removeClass('ev-tab-active').find('.ev-module-edit-box:first').attr('data-loadanimate', 0);
                        curDom.find('.video-module, .ev-custom-video').each(function(i, dom){
                            var video = $(dom).find('video');
                            video.length && video[0].pause();
                        });
                        curDom.find('.ev-audio-module').each(function(i, dom){
                            dom = $(dom);
                            if(dom.find('.playing').length){
                                dom.find('.playing').trigger('click');
                            }
                        });
                    }
                    if (curDom.attr('data-tabmark') === tabmark) {
                        curDom.removeClass('none').addClass('ev-tab-active');
                        var dom = curDom.find('.ev-module-edit-box:first');
                        if (dom.data('animate-name')) {
                            wsf.f.animationLoadMo(dom);
                        }else{
                            dom.find('.ev-module-edit-box,.customModule').each(function (i, dom) {
                                dom = $(dom);
                                if(dom.hasClass('ev-module-edit-box')){
                                    var videoModule = dom.find('div.ev-custom-video');
                                    if(videoModule.length && !videoModule.data('loaded')){
                                        videoModule.data('loaded', 1)
                                        wsf.videoModule.load(videoModule);
                                    }
                                    var audioModule = dom.find('div.ev-audio-module');
                                    if(videoModule.length && !audioModule.data('loaded')){
                                        audioModule.data('loaded', 1)
                                        wsf.audioModule.load(audioModule);
                                    }
                                    var progressModule = dom.find('div.ev-progress-module');
                                    if(progressModule.length && !progressModule.data('loaded')){
                                        progressModule.data('loaded', 1)
                                        wsf.f.progressLoading(progressModule);
                                    }
                                }
                                // 关于视频的加载
                                dom.hasClass('customModule') && (function(){
                                    var videoModule = dom.find('div.video-module');
                                    if(videoModule.length){
                                        wsf.videoModule.load(videoModule);
                                    }
                                })();
                                if(dom.data('animate-name')){
                                    wsf.f.animationLoadMo(dom);
                                }else{
                                    wsf.f.imgLazyLoading(dom);
                                }
                            });
                        }
                    }
                }
            });
        }
    },
    hideFun: function(targetDom, targetData){
        var one, two;
        var CModulePA = targetDom.closest('.CModulePA');
        var classMark = '';
        switch(targetData.big*1){
            case 9:
                classMark = 'tab';
            break;
            case 26:
                classMark = 'interact';
            break;
        }
        switch(targetData.threetype){
            case 1:
                one = targetDom;
                two = CModulePA.children('.ev-'+ classMark +'-container-two[data-groupmark="'+ targetData.groupmark +'"][data-tabmark="'+ targetData.tabmark +'"]');
                break;
            case 2:
                two = targetDom;
                one = CModulePA.children('.ev-'+ classMark +'-container-one[data-groupmark="'+ targetData.groupmark +'"][data-tabmark="'+ targetData.tabmark +'"]');
                break;
        }
        var hideTime = setTimeout(function(){
            one.find('.ev-module-edit-box').removeClass('ev-tab-active');
            two.addClass('none').removeClass('ev-tab-active');
            if(targetData.big * 1 == 26){
                two.children('.ev-module-edit-box').attr('data-loadanimate', 0);
                if(targetData.small == 2 && one.data('triggerShow') == 0){
                    one.removeClass('edit-hidden');
                    if (one.children('.ev-module-edit-box').data('animate-name')) {
                        wsf.f.animationLoadMo(one.children('.ev-module-edit-box'));
                    }
                }
            }
        }, 80);
        one.data('hide-time', hideTime);
    },
    loadBind: function(){
        var that = this;
        wsf.pDom.bodyDom.on(
            'mouseenter.tabSwitch click.tabSwitch mouseleave.tabSwitch',
            '.ev-tab-container-one, .ev-tab-container-two, .ev-interact-container-one, .ev-interact-container-two',
            function (e) {
                var $this = $(this);
                var $thisData = $this.data();
                var events = e;
                var CModulePA = $this.closest('.CModulePA');
                var oneDom = null;
                if($thisData.threetype == 1 && $thisData.trigger == events.type && (($thisData.big == 26 && !is_action) || $thisData.big != 26)){
                    that.swtichFun($this, $thisData);
                    return;
                }
                if(($thisData.big == 26 && !is_action) || ($thisData.big == 9 && $thisData.small == 3) && events.type == "mouseenter"){
                    if($thisData.threetype == 1){
                        clearTimeout($this.data('hide-time'));
                    }else{
                        if($thisData.big == 26){
                            oneDom = CModulePA.children('.ev-interact-container-one[data-groupmark="'+ $thisData.groupmark +'"][data-tabmark="'+ $thisData.tabmark +'"]');
                        }else if($thisData.big == 9){
                            oneDom = CModulePA.children('.ev-tab-container-one[data-groupmark="'+ $thisData.groupmark +'"][data-tabmark="'+ $thisData.tabmark +'"]');
                        }
                        clearTimeout(oneDom.data('hide-time'));
                    }
                }
                if(events.type == "mouseleave" && ($thisData.small == 3 || $thisData.big == 26)){
                    !is_action && that.hideFun($this, $thisData);
                }
            }
        );
    }
};
/*弹出层行的js*/
wsf.evAlertRow = {
    rowDom: null,
    writeStyle: function (cSId, styleArray) {
        var cSh = "";
        $.each(styleArray, function(i, v){
            $.each(v, function(key, v_){
                v_ = v_.replace(/\\\'/g, "'");
                cSh += v_;
            });
        });
        $("head").append("<style type='text/css' id=alert_" + cSId + ">" + cSh + "</style>");
    },
    alertJson: {},
    upMocon: function(curAlertDom, data){
        if(data.SMModuleConfigJson){
            $.each(data.SMModuleConfigJson, function(k, v){
                moduleConfig[k] = v;
            });
        }
        curAlertDom.find('.ev-module-edit,.customModule').each(function (i, dom) {
            dom = $(dom);
            var id = null, z = null;
            if (dom.hasClass('customModule')) {
                
                //收藏点赞绑定事件功能
                id = dom.children('.Mo').attr('id');
                z = wsf.f.s_j(dom.attr('data-attr')).z;
                var moduleData = wsf.f.s_j(dom.attr('data-attr'));
                (function () {
                    if (moduleData.mt == 21) {
                        CatalogMore = 1;
                        var menuCatalogMoreModule = $('#menuCatalogMoreModule');
                        !menuCatalogMoreModule.length && (menuCatalogMoreModule = $('<div id="menuCatalogMoreModule" class="menu-catalog-more-module"></div>').appendTo($('body')));
                        menuCatalogMoreModule.append(data.typeOtherHtml);
                    }
                    if(moduleData.mt == 8){
                        userUserLoginInfo();
                    }
                })();
                if(moduleData.mt == 7){
                    wsf.f.upJsFun(dom, data.MFormData);
                }else{
                    wsf.f.upJsFun(dom);
                }
            } else {
                wsf.f.upJsFun(dom);
            }
            if (data.jsData) {
                wsf.queryMJsEffect(data.jsData);
            }
        });
    },
    showAlert: function (alertId, openType) {//显示弹窗函数
        var that = this,
            jsonCurDom = null,
            url = null,
            parWin = window.parent || null;
        that.rowDom.is(':hidden') && that.rowDom.show();
        jsonCurDom = that.alertJson[alertId];
        var setTimFun = function(jsonCurDom){//延迟动画加载函数
            setTimeout(function () {
                var curDom = jsonCurDom[0],
                    curDomEditBox = curDom.find('div.ev-module-edit-box:first'),
                    aDelay = 0,aDuration = 0, timeout;
                that.rowDom.is(':hidden') && that.rowDom.show();
                jsonCurDom[0].removeClass('none');
                that.rowDom.append(jsonCurDom[1].removeClass('none'));
                if (curDomEditBox.data('animate-name')) {
                    wsf.f.animationLoadMo(curDomEditBox);
                }else{
                    curDomEditBox.find('.ev-module-edit,.customModule').each(function (i, dom) {
                        dom = $(dom);
                        if(dom.hasClass('customModule')){
                            wsf.f.animationLoadMo(dom);
                        }else{
                            var curDomEditBox = dom.find('div.ev-module-edit-box:first');
                            if(curDomEditBox.data('animate-name')){
                                wsf.f.animationLoadMo(curDomEditBox);
                            }
                        }
                        wsf.f.imgLazyLoading(dom);
                    });
                }
            }, openType === 'load' ? jsonCurDom[0].attr('data-opendelay') * 1000 : 0);
        };
        if (!jsonCurDom) {
            $.get('/Ajax/TJIndex.php?type=8&username=' + user_name + '&tab_id=' + alertId + '&ch_id=' + GData.allChId + (is_action ? '&is_design=1' : ''), function (data) {
                if(data.success * 1){
                    var loadFun = function(){
                        jsonCurDom = that.alertJson[alertId] = [$(data.html.replace(/\\'/g,"'")), $('<div class="none alertBg alertBg-' + alertId + '" id="alertBg_' + alertId + '"></div>')];
                        that.rowDom.find('.CModulePA').append(jsonCurDom[0]);
                        //回显点赞,收藏方法
                        if(data.docSysIds || data.SMAlbumSysIds || data.SMAlbumIds){
                            GData.docSysIds = GData.SMAlbumSysIds = GData.SMAlbumIds = '';
                            data.docSysIds  && (GData.docSysIds = data.docSysIds);
                            data.SMAlbumSysIds && (GData.SMAlbumSysIds = data.SMAlbumSysIds);
                            data.SMAlbumIds  && (GData.SMAlbumIds = data.SMAlbumIds);
                            wsf.albumsDataLoad();
                        }
                        if(jsonCurDom[0].attr('data-positiontype') * 1){
                            wsf.f.positionShow(jsonCurDom[0], jsonCurDom[0].attr('data-position') - 1);
                        }else{
                            jsonCurDom[0].css({
                                left: Math.floor(jsonCurDom[0].attr('data-lshifting') * 1 + (wsf.pDom.bodyDom.width() - userSiteWidth) / 2) + 'px',
                                top: jsonCurDom[0].attr('data-tshifting') + 'px'
                            });
                        }
                        that.upMocon(jsonCurDom[0], data);
                        setTimFun(jsonCurDom);
                        if(!is_action){
                            wsf.evAlertRow.writeStyle(alertId, [data.Mstyle, data.SMstyle]);
                        }
                    };
                    if(data.MShowStyleList[13] || data.MShowStyleList[12] || data.MShowStyleList[9] || data.MShowStyleList[7]){
                        var whenArray = [];
                        if(data.MShowStyleList[12] && typeof(BMap) === 'undefined'){
                            /* whenArray.push($.ajax({
                                dataType:'script',
                                scriptCharset:'utf-8',
                                url:'//api.map.baidu.com/getscript?v=2.0&ak='+ GData.BAI_DU_MAP_AK+'&services=&t=20180823194355'
                            })); */
                            // console.log(88888);
                            // $( "<script>", {
                            //     dataType:'text/javascript',
                            //     scriptCharset:'utf-8',
                            //     src: '//api.map.baidu.com/getscript?v=2.0&ak='+ GData.BAI_DU_MAP_AK+'&services=&t=20180823194355'
                            // }).appendTo( "head" );
                            whenArray.push($.getScript('//api.map.baidu.com/getscript?v=3.0&ak='+ GData.BAI_DU_MAP_AK+'&services=&t=20180823194355'));
                            whenArray.push($.getScript('//api.map.baidu.com/library/SearchInfoWindow/1.5/src/SearchInfoWindow_min.js'));
                        }
                        if(data.MShowStyleList[13] && typeof(dhtmlXChart) === 'undefined'){
                            whenArray.push($.getScript(ALIYUN_OSS_DOMAIN + 'plugins/public/dhtmlxChart/dhtmlxchart.min.js'));
                        }
                        if(data.MShowStyleList[9] && typeof(chplayer) === 'undefined'){
                            whenArray.push($.ajax({
                                dataType:'script',
                                scriptCharset:'utf-8',
                                url:ALIYUN_OSS_DOMAIN + 'plugins/public/js/chplayer/chplayer.min.js'
                            }));
                        }
                        if(data.MShowStyleList[7]){
                            if(typeof(tncode) === 'undefined'){
                                !$('#sliding_css_src').length && $( "<link/>", {
                                    rel: "stylesheet",
                                    type: "text/css",
                                    id: 'sliding_css_src',
                                    href: ALIYUN_OSS_DOMAIN + "plugins/public/js/slidingVerification/sliding_verification_style.css"
                                }).appendTo( "head" );
                                whenArray.push($.getScript(ALIYUN_OSS_DOMAIN + 'plugins/public/js/slidingVerification/sliding_tn_code.js'));
                            }
                            if(typeof(jQuery.validationEngine) === 'undefined'){
                                !$('#from_module_form_css').length && $( "<link/>", {
                                    rel: "stylesheet",
                                    type: "text/css",
                                    id: 'from_module_form_css',
                                    href: ALIYUN_OSS_DOMAIN + "plugins/public/js/from/form.min.css"
                                }).appendTo( "head" );
                                whenArray.push($.getScript(ALIYUN_OSS_DOMAIN + 'plugins/public/js/from/form.min.js'));
                            }
                        }
                        if(whenArray.length){
                            $.when.apply(this, whenArray).done(function(){
                                loadFun();
                            });
                        }else{
                            loadFun();
                        }
                    }else{
                        loadFun();
                    }
                }else{
                    // wsf.f.alertWindow(data.error, 'error', 3);
                }
            }, 'json');
        }else{
            setTimFun(jsonCurDom);
        }
    },
    loadBind: function () {
        var that = this;
        that.rowDom = $('div.alertPublicModuleRow:first');
        wsf.pDom.bodyDom.on({
            click: function () {
                var $this                = $(this),
                    evAlertContainer     = $this.closest('.ev-alert-container'),
                    alertPublicModuleRow = $this.closest('.alertPublicModuleRow'),
                    tabStrId             = evAlertContainer.attr('id').substr(5),
                    closetype            = parseInt(evAlertContainer.attr('data-closetype')),
                    opentype             = parseInt(evAlertContainer.attr('data-opentype'));

                if (evAlertContainer.attr('data-new') * 1 === 1) {
                    // DF.pub.Fun.addAlertContainer.editName();
                } else {
                    if (opentype === 0 && closetype === 1 && !is_action) {
                        var alertTabList = {},
                            alertTabCookieInfo = readCookie(user_name+'_tjAlertTabList');

                        if (alertTabCookieInfo) {
                            alertTabList = JSON.parse(alertTabCookieInfo);
                        }

                        alertTabList[tabStrId] = new Date().getTime();
                        writeCookie(user_name+'_tjAlertTabList', JSON.stringify(alertTabList), 1000*24);
                    }

                    evAlertContainer.addClass('none');
                    evAlertContainer.find('.ev-module-edit-box').each(function(i, dom){
                        dom = $(dom);
                        if(dom.attr('data-animate-name')){
                            dom.addClass('load-animate');
                            dom.removeData('loadanimate');
                            dom.attr('data-loadanimate', 0);
                        }
                    })
                    alertPublicModuleRow.children('#alertBg_' + tabStrId).addClass('none');
                    is_action && evAlertContainer.removeClass('curEditModuleSize');
                    if (!evAlertContainer.siblings('.ev-alert-container:not(.none)').length) {
                        alertPublicModuleRow.removeClass('editShowAlertRow');
                    }
                }
                if(is_action){
                    wsf.pDom.bodyDom.trigger('click');
                    parWin.Ev.admin.tj.v.editShowAlert = null;
                }
                evAlertContainer.find('.video-module, .ev-custom-video').each(function(i, dom){
                    var video = $(dom).find('video');
                    video.length && video[0].pause();
                });
                evAlertContainer.find('.ev-audio-module').each(function(i, dom){
                    dom = $(dom);
                    if(dom.find('.playing').length){
                        dom.find('.playing').trigger('click');
                    }
                });
                !is_action && evAlertContainer.find('.online-video-module').each(function(i, dom){
                    dom = $(dom);
                    var iframe = dom.find('iframe');
                    var video = dom.find('video');
                    if(iframe.length){
                        iframe.attr('data-src', iframe.attr('src'));
                        iframe.removeAttr('src');
                        dom.attr('data-removesrc',1);
                    }
                    video.length && video[0].pause();
                });
            }
        }, '.alert-close-btn');
        if (!is_action && GData.tabAlertStrMs) {
            $.each(GData.tabAlertStrMs.split(','), function (i, v) {
                that.showAlert(v, 'load');
            });
        }
    }
};
/*进度变换*/
wsf.progress = {
    judge: function(){
        var that = this;
        that.defDomJson = [];
        $('div.ev-progress-module').each(function(i, dom){
            dom = $(dom);
            if(!dom.closest('.load-animate').length){
                that.defDomJson.push({
                    'dom': dom,
                    'oTop': dom.offset().top,
                    'loaded': false
                });
            }
        });
    },
    showProgress: function(t, h){
        var that = this;
        $.each(that.defDomJson, function (i, v) {
            if (!v.loaded) {
                if ((t + h * 0.9) >= v.oTop) {
                    that.load(v.dom);
                    v.loaded = true;
                }
            }
        });
    },
    numberSkin: {
        skin2:{type:'img', url:'img'},
        skin2:{type:'span', cname:'img'}
    },
    barFun: {
        type1:function(){
            return null;
        },
        type2: function (dom) {
            var bar = dom.find('.bar'),
                maxVal = dom.attr('data-maxval') * 1;
            return function (v) {
                bar.css({width: (v / maxVal) * 100 + '%'});
            };
        },
        type3: function (dom) {
            var bar = dom.find('.bar'),
                maxVal = dom.attr('data-maxval') * 1;
            return function (v) {
                bar.css({width: (v / maxVal) * 100 + '%'});
            };
        },
        type4: function (dom) {
            var bar = dom.find('.bar'),
                maxVal = dom.attr('data-maxval') * 1;
            return function (v) {
                bar.css({height: (v / maxVal) * 100 + '%'});
            };
        },
        type5: function (dom) {
            var bar = dom.find('.bar'),
                maxVal = dom.attr('data-maxval') * 1;
            return function (v) {
                bar.css({height: (v / maxVal) * 100 + '%'});
            };
        }
    },
    progressArr:{},
    load: function (progress) {
        var that = this,
            smallType = (function(){
                var barClass = progress.find('div:first-child').attr('class'),
                    x = 0,
                    arr = ['p-num','p-bar-2','p-bar-3','p-bar-4','p-bar-5'];
                $.each(arr, function(i, v){
                    if(barClass.indexOf(v) !== -1){
                        x = i + 1;
                        return false;
                    }
                });
                return x;
            }()),
            j = {
                target: progress.find('.num-e'),
                startVal: progress.attr('data-startval') * 1,
                endVal: progress.attr('data-endval') * 1,
                decimals: progress.attr('data-decimals') * 1,
                duration: progress.attr('data-duration') * 1,
                grouping: !!(progress.attr('data-grouping') * 1),
                easingFn: 'easeOut',
                separator: ',',
                decimal: '.',
                prefix: '',
                suffix: '',
                numerals: [],
                runCallback: that.barFun['type' + smallType](progress),
                startCallback: function () {},
                endCallback: function () {}
            };
        ($.inArray(smallType, [2, 3, 4, 5]) !== -1) && progress.find('.num-m').text(progress.attr('data-maxval'));
        if(j.duration){
            progress.countUp(j);
            progress.data('countup').start();
        }
    }
};
/* 更新会员，浏览量数据 */
wsf.upUUPV = function(){
    var moduleArr = $('.ev-module-edit[data-big="23"]');
    var ids = (function(){
        var arr = [];
        moduleArr.each(function(i, dom){
            arr.push($(dom).attr('id').substr(5))
        });
        return arr;
    })();
    $.ajax({
        'url': '/Ajax/TJIndex.php?username='+user_name+'&type=9&m_ids='+ ids.join(),
        type: "POST",
        async: false,
        cache: false,
        dataType: 'json',
        success: function (data) {
            if(data.code == 200){
                var list = data.list.list;
                moduleArr.each(function(i, dom){
                    var dom = $(dom);
                    var id = dom.attr('id').substr(5);
                    if(list[id] != undefined){
                        dom.find('.ev-progress-module').attr('data-endval',list[id]).data('endval', list[id]).find('.num-e').text(list[id]);
                    }
                });
            }
            
        }
    });
};
/**
 * 页面城市选择模块
 */
wsf.cityModule = {
    currentLang: null,
    list: null,
    renderList: function(dom, mark){
        var that = this;
        if(!dom.data('renderList') || mark){
            var valDom = dom.find('.current-city .val');
            var domId = dom.closest('.ev-module-edit').attr('id');
            var domConfig = moduleConfig[domId];
            var cityList = [];
            if(dom.find('.ev-city-popup').length){
                dom.find('.ev-city-popup').remove();
            }
            if(domConfig.cityList){
                $.each(domConfig.cityList, function(key, v){
                    cityList.push(v);
                });
            }
            cityList.sort(function(a,b) {
                return a.f.charCodeAt() - b.f.charCodeAt();
            });
            var letterList = cityList.map(function(v){
                return v.f;
            });
            var newLetterList = letterList.filter(function(item, index, curArr){
                return curArr.indexOf(item, 0) === index;
            });
            var headHTMLArray = [
                '<div class="m-c-t">',
                    '<h2 class="m-c-t-l">全国</h2>',
                    '<div class="m-c-t-r">',
                        '<div class="m-c-s focus">',
                            '<span class="m-c-s-icon evIcon evIcon-search-6"></span>',
                            '<input type="text" class="m-c-k" />',
                        '</div>',
                        '<span class="m-c-c-icon evIcon evIcon-mult-8"></span>',
                    '</div>',
                '</div>',
            ];

            var cityHTMLArray = [];
            var letterHTMLArray = [];
            
            
            // 组装字母
            letterHTMLArray.push(
            '<div class="c-l-w">',
                '<div class="c-l-w-i">',
                    '<ol class="c-l">'
            );
            
            newLetterList.forEach(function(v){
                letterHTMLArray.push(
                    '<li class="c-l-e">',
                        '<a class="c-l-e-i" data-target="'+ v +'" href="javascript:;">'+ v +'</a>',
                    '</li>'
                );
            });

            letterHTMLArray.push(
                    '</ol>',
                '</div>',
            '</div>'
            );

            // 组装省市列表
            cityList.forEach(function(cv, ci){
                var cityArray = [];
                // 判断有没有市，并且组装
                if(cv.cl && cv.cl.length){
                    // 组装市
                    cityArray.push(
                        '<dd class="t-l-b">',
                            '<div class="t-l-b-i">',
                                '<ol class="t-l-b-l">'
                    );
                    cv.cl.forEach(function(sV, sI){
                        var target = parseInt(sV.t) === 1 ? 'target="_blank"' : '';
                        var href = sV.l || 'javascript:;';
                        cityArray.push(
                            '<li class="t-l-b-l-e">',
                                '<a class="t-l-b-l-e-i" href="'+ href +'" '+ target +'>'+ sV.n +'</a>',
                            '</li>'
                        )
                    });
                    cityArray.push(
                                '</ol>',
                            '</div>',
                        '</dd>'
                    );
                }
                cityHTMLArray.push(
                    '<div class="t-l" data-letter="'+ cv.f +'">',
                        '<dl class="t-l-i">',
                            '<dt class="t-l-h">',
                                '<div class="t-l-h-i">',
                                    '<strong class="t-l-h-t">',
                                        '<a class="t-l-h-t-i" href="#">'+ cv.n +'</a>',
                                    '</strong>',
                                '</div>',
                            '</dt>',
                            cityArray.join(''),
                        '</dl>',
                    '</div>'
                )
            });

            var htmlArray = [
                '<div class="ev-city-popup">',
                    '<div class="m-c-m-l">',
                        headHTMLArray.join(''),
                        letterHTMLArray.join(''),
                        '<div class="t-l-w">',
                            '<div class="t-l-w-i">',
                                cityHTMLArray.join(''),
                            '</div>',
                        '</div>',
                    '</div>',
                '</div>'
            ];
            dom.data('renderList', 1).append(htmlArray.join(''));
            // dom.append(htmlArray.join(''));
        }
    },
    load: function(obj){
        var that = this;
        var cityModule = obj || $('.ev-city-module');
        if(cityModule.length){
            cityModule.each(function(i, dom){
                var $dom = $(dom);
                var valDom = $dom.find('.current-city .val');
                var domId = $dom.closest('.ev-module-edit').attr('id');
                var domConfig = moduleConfig[domId];
                var trigger = domConfig.trigger;
                $dom.on(trigger, '.current-city', function(){
                    $dom.addClass('open');
                    that.renderList($dom);
                });
                $dom.on('mouseleave', function(){
                    $dom.removeClass('open');
                });
                $dom.on('click', '.m-c-c-icon', function(){
                    $dom.removeClass('open');
                });
                $dom.on('click', '.c-l-e-i', function(){
                    var $this = $(this);
                    var target = $this.data('target');
                    var TLW = $dom.find('.t-l-w');
                    $dom.find('.t-l[data-letter="'+ target +'"]').each(function(index, tlDom){
                        tlDom = $(tlDom);
                        if(index == 0){
                            TLW.scrollTop(TLW.scrollTop() + tlDom.position().top);
                        }
                        tlDom.addClass('show-blink');
                        setTimeout(function(){
                            tlDom.removeClass('show-blink');
                        }, 2000);
                    });
                });
                $dom.on('change', '.m-c-k', function(){
                    var keyWord = $(this).val();
                    if(keyWord){
                        $dom.find('.t-l').each(function(){
                            var $this = $(this);
                            if($this.text().indexOf(keyWord) != -1){
                                $this.show();
                            }else{
                                $this.hide();
                            }
                        });
                    }else{
                        $dom.find('.t-l').show();
                    }
                });
                $dom.on('click', '.t-l-b-l-e-i', function(){
                    var text = $(this).text();
                    sessionStorage.setItem('currentCity', $(this).text());
                    valDom.text(text);
                    $dom.removeClass('open');
                });
                if(sessionStorage.getItem('currentCity')){
                    valDom.text(sessionStorage.getItem('currentCity'));
                }
            });
        }
    }
};
/**
 * 页面翻译模块
 */
wsf.translatorModule = {
    currentLang: null,
    list: [],
    renderList: function(dom){
        var that = this;
        if(!dom.data('renderList') || is_action){
            that.list = window.$pTranslateLanguageJson;
            var li_arr = [];
            var valDom = dom.find('.current-language .val');
            var domId = dom.closest('.ev-module-edit').attr('id');
            var domConfig = moduleConfig[domId];
            var langList = domConfig.langList;
            
            var newList = that.list.filter(function(curV){
                return $.inArray(curV.value, langList) !== -1;
            });
            newList.forEach(function(v, i){
                var active = '';
                if(that.currentLang == v.value){
                    active='active';
                    valDom.html(v.alias);
                }
                li_arr.push('<li class="l-item '+ active +'" data-value="'+ v.value +'">'+ v.alias +'</li>');
            });
            dom.data('renderList', 1).find('.language-list').html(li_arr.join(''));
        }
    },
    load: function(){
        var that = this;
        var translatorModule = $('.ev-translator-module');
        that.list = window.$pTranslateLanguageJson;
        if(window.translate){
            // 设置默认翻译为的语种
            translate.language.setDefaultTo(GData.defaultLang || 'chinese_simplified');
            // 监控页面动态渲染的文本进行自动翻译
            translate.listener.start();
            // 是否显示 select选择语言的选择框，true显示； false不显示。默认为true
            translate.selectLanguageTag.show = false; 
            // 设定切换语言所支持的语种 (中文简体，繁體中文，英文 ，日語，韩文，西班牙语，阿拉伯语，法语，俄语)
            translate.selectLanguageTag.languages = 'chinese_simplified,chinese_traditional,english,korean,spanish,arabic,french,russian,deutsch';
            translate.service.use('client.edge');
            translate.ignore.class.push('no-translate');
            translate.execute();
            // 获取当前显示的是什么语种
            that.currentLang = translate.language.getCurrent();

            if (translatorModule.length) {
                translatorModule.each(function(i, dom){
                    var $dom = $(dom);
                    var valDom = $dom.find('.current-language .val');
                    var domId = $dom.closest('.ev-module-edit').attr('id');
                    var domConfig = moduleConfig[domId];
                    var trigger = domConfig.trigger;
                    $dom.on(trigger, function(){
                        $dom.addClass('open');
                        that.renderList($dom);
                    });
                    $dom.on('mouseleave', function(){
                        $dom.removeClass('open');
                    });
                    $dom.on('click', '.l-item', function(){
                        var $this = $(this);
                        var val = $this.data('value');
                        if(!$this.hasClass('active') && val){
                            translate.changeLanguage(val);
                            $this.addClass('active');
                            $this.siblings('li').removeClass('active');
                            $dom.removeClass('open');
                            valDom.html($this.text());
                        }
                    });
                    that.list.forEach(function(v, i){
                        if(that.currentLang == v.value){
                            valDom.html(v.alias);
                        }
                    });
                });
            }


        }
    }
};

/*
    show-big-pic
    createDate:2017/09/24
    产品最终页的图片效果
*/
wsf.innerPreview = function (id) {
    var preview = $(id),
        bigPics = preview.find(".show-big-pic"),
        bigPic = bigPics.find(".pic"),
        bigPicsImg = bigPic.find("img"),
        bigPicW = parseInt(bigPic.width()),
        bigPicH = parseInt(bigPic.height()),
        smallListArea = preview.find(".small-pic-list-area"),
        sList = smallListArea.find(".small-pic-list"),
        sListW = sList.width(),
        sListUl = sList.find("ul"),
        ulW = 0,
        spic = null,
        zoom = null;
    sListUl.find("li").each(function (li) {
        li = $(this);
        ulW += li.width();
    });
    sListUl.width(ulW).attr('data-left', 0);
    // 左右按钮绑定事件
    smallListArea.on({
        click: function () {
            var $this = $(this),
                action = $(this).data('action'),
                curleft = sListUl.attr('data-left') * 1,
                moveleft = null;
            if (!$this.data('click') && ulW > sListW) {
                $this.data('click', 1);
                switch (action) {
                    case 'left':
                        curleft != 0 && (function () {
                            if (curleft - sListW != 0) {
                                moveleft = Math.min(Math.max(curleft - sListW, curleft), sListW);
                            } else {
                                moveleft = sListW;
                            }
                            moveleft = -moveleft;
                        })();
                        break;
                    case 'right':
                        (function () {
                            moveleft = Math.min(ulW - curleft - sListW, sListW);
                        })();
                        break;
                }
                sListUl.animate({
                    marginLeft: -(curleft + moveleft)
                }, 600, function () {
                    sListUl.attr('data-left', (curleft + moveleft));
                    $this.data('click', 0);
                });
            }
        }
    }, '.move-btn');
    // 小图li绑定事件
    smallListArea.on({
        mouseenter: function () {
            var $this = $(this);
            $this.addClass("liCur").siblings('.liCur').removeClass('liCur');
            var sImgUrl = $this.find("img").attr('data-imgsrc'),
                oImg = new Image();
            sImgUrl = sImgUrl.indexOf('?') != '-1' ? sImgUrl.substr(0, sImgUrl.indexOf('?')) : sImgUrl;
            bigPicsImg.attr('src', '/images/VNew/loading-wait-bg.gif').attr('data-imgsrc', sImgUrl);
            oImg.onload = function () {
                bigPicsImg.attr({
                    "src": oImg.src
                });
            }
            oImg.src = sImgUrl + '?x-oss-process=image/resize,m_lfit,limit_1,w_' + parseInt(bigPicW) + ',h_' + parseInt(bigPicH);
        }
    }, 'li');
    // 初始化方法
    smallListArea.find('li').each(function (i, dom) {
        dom = $(dom);
        var pic = dom.find('.pic'),
            getImage = pic.find('img'),
            sImgUrl = getImage.attr('data-imgsrc'),
            oImage = new Image();
        sImgUrl = sImgUrl.indexOf('?') != '-1' ? sImgUrl.substr(0, sImgUrl.indexOf('?')) : sImgUrl;
        getImage.attr('data-imgsrc', sImgUrl);
        oImage.onload = function () {
            getImage.attr('src', this.src);
        };
        oImage.src = sImgUrl + '?x-oss-process=image/resize,m_lfit,limit_1,w_' + pic.width().toFixed(0) + ',h_' + pic.height().toFixed(0);
        i == 0 && dom.trigger('mouseenter');
    });
    if (!bigPics.data('desabledZoom')) {
        bigPics.on({
            mouseenter: function (event) {
                if(!bigPics.data('videoplay')){
                    var $this = $(this),
                        getImgSrc = $this.find("img").attr('data-imgsrc');
                    if ($this.data("hover") != 1) {
                        $this.data("hover", 1);
                        zoom = $('<div class="zoom"></div>');
                        $this.find('.pics').append(zoom);
                        spic = $('<div class="zoomPic"><img src="' + getImgSrc + '" /></div>');
                        spic.css({
                            "left": (preview.width() + 10) + "px",
                            "top": "0px"
                        });
                        spic.appendTo(preview);
                    }
                    var l = event.clientX,
                        t = event.clientY + $(document).scrollTop();
                    fun(l, t);
                }
            },
            mousemove: function (event) {
                zoom && fun(event.clientX, event.clientY + $(document).scrollTop());
            },
            mouseleave: function () {
                var $this = $(this);
                $this.removeData("hover");
                zoom.remove();
                spic.remove();
                zoom = spic = null;
            }
        });
        var fun = function (l, t) {
            var zoomW = (zoom.outerWidth() || zoom.width());
            zoomH = (zoom.outerHeight() || zoom.height());
            l = l - bigPic.offset().left - zoomW / 2;
            t = t - bigPic.offset().top - zoomH / 2;
            if (l < 0) {
                l = 0;
            } else if (l > bigPic.outerWidth() - zoomW) {
                l = bigPic.outerWidth() - zoomW;
            }
            if (t < 0) {
                t = 0;
            } else if (t > bigPic.outerHeight() - zoomH) {
                t = bigPic.outerHeight() - zoomH;
            }
            zoom.css({
                "left": l + "px",
                "top": t + "px"
            });
            var lx = l / (bigPic.outerWidth() - zoomW);
            var tx = t / (bigPic.outerHeight() - zoomH);
            var img = spic.children("img");
            img.css({
                "left": -lx * (img.outerWidth() - spic.outerWidth()) + "px",
                "top": -tx * (img.outerHeight() - spic.outerHeight()) + "px"
            });
        };
    }
    bigPics.on({
        click: function(e){
            var clickDom = $(e.target),
                mp4VideoDiv = bigPics.find('.mp4-video-div');
            switch(true){
                case clickDom.hasClass('mp4-play'):
                    bigPics.trigger('mouseleave');
                    mp4VideoDiv.show().find('#prodectVideo').css({
                        width: bigPics.width() + 'px',
                        height: bigPics.height() + 'px'
                    });
                    new chplayer({
                        container: '#prodectVideo',
                        variable: 'player',
                        autoplay: true,
                        poster: mp4VideoDiv.attr('data-pic'), //封面图片地址
                        video: mp4VideoDiv.attr('data-url')
                    });
                    bigPics.data('videoplay',1).find('.mp4-close').show();
                    clickDom.hide();
                    break;
                case clickDom.hasClass('mp4-close'):
                    mp4VideoDiv.hide().find('#prodectVideo').html('');
                    clickDom.hide();
                    bigPics.removeData('videoplay').find('.mp4-play').show();
                    bigPics.trigger('mouseenter');
                    break;
            }
        }
    },'.mp4-play,.mp4-close');
    //增加全屏预览效果
    bigPicsImg.viewer({
        url: "data-imgsrc",
        inline: false,
        viewed: function() {
            bigPicsImg.viewer('zoomTo', 1);
        }
    });
    // smallListArea.viewer({
    //     url: "data-imgsrc"
    // });
};
/*
    pWapBuyArea
    createDate:2017/09/24
    产品最终页的手机扫码购买功能
*/
wsf.innerWapBuy = function () {
    var pWapBuyArea = $("#pWapBuyArea");
    if (pWapBuyArea.length) {
        var pPriceList = $("#pPriceList"),
            t;
        if (pPriceList.length) {
            t = pPriceList.position().top;
        } else {
            t = 32;
        }
        pWapBuyArea.css({
            top: t + 'px'
        });
        pWapBuyArea.on({
            mouseenter: function () {
                $(this).addClass('p-wap-buy-area-hover');
            },
            mouseleave: function () {
                $(this).removeClass('p-wap-buy-area-hover');
            }
        });
    }
};
/*产品选号*/
wsf.tel_select = function(pro_id, v_id){
    if(v_id){
        var v = $('#'+v_id).val() || '';
    }
    wsf.f.openWin('选号', '/dom/phone_select.php?username=' + user_name + '&pid='+pro_id+'&v='+v, 850, 500);
};
/*注册登录协议窗口*/
wsf.agreement_alert = function(url, title){
    wsf.f.openWin(title, url, 850, 500);
};
/*
    pCouponList
    createDate:2018/07/16
    优惠券领取弹窗
*/
wsf.pShowGetCoupon = function(){
    var pCouponList = $('#pCouponList');
    if (pCouponList.length) {
        pCouponList.on('click', '.get-coupon-b', function () {
            var zzUserid = readCookie('zz_userid');

            if (!zzUserid) {
                pc_login_alert();
                return true;
            }

            var id = parseInt($(this).data("id"));
            if (!id || !pro_id) {
                wsf.f.alertWindow("参数有误!");
                return true;
            }

            var jiFenObj = $("#pCouponList");
            var payJiFen = parseInt($(this).data("payjifen"));
            var interactionJiFen = parseInt($(this).data("interactionjifen"));
            var payJiFenCount = parseInt(jiFenObj.attr("data-payjifen"));
            var payJifFenName = jiFenObj.attr("data-payjifenname");
            var interactionJiFenCount = parseInt(jiFenObj.attr("data-interactionjifen"));
            var interactionJifFenName = jiFenObj.attr("data-interactionjifenname");

            var checkPay = 1;
            var checkInteraction = 1;

            if (payJiFen && payJiFen > payJiFenCount) {
                checkPay = 0;
            }
            if (interactionJiFen && interactionJiFen > interactionJiFenCount) {
                checkInteraction = 0;
            }

            if (checkInteraction === 0 || checkPay === 0) {
                var errorMsg = '领取优惠卷需要';
                if (payJiFen) {
                    errorMsg += payJifFenName + payJiFen + '积分,<br />';
                }
                if (interactionJiFen) {
                    errorMsg += interactionJifFenName + interactionJiFen + '积分,';
                }

                if (checkInteraction === 0 && checkPay === 0) {
                    errorMsg += "您的"+payJifFenName+'积分和'+interactionJifFenName+'不足,';
                } else if (checkInteraction === 0) {
                    errorMsg += "您的"+interactionJifFenName+'不足,<br />';
                } else if (checkPay === 0) {
                    errorMsg += "您的"+payJifFenName+'积分不足,';
                }
                errorMsg += '暂不能参加该活动,非常抱歉!';

                wsf.f.alertWindow(errorMsg, 'error');
                return true;
            }

            var yes = function ()
            {
                $.getJSON("/Ajax/Product.php?type=11&claim_type=1&username=" + user_name + "&pid=" + pro_id + "&coupon_id=" + id, function (data) {
                    var pdom = null,
                        idom = '<i class="icon fail-icon"></i>',
                        htmls;

                    data.status = parseInt(data.status);
                    if (parseInt(data.success) === 0  && data.status === 0) {
                        wsf.f.alertWindow(data.error);
                        return true;
                    }

                    switch (data.status * 1) {
                        case 0:
                            pdom = '<div class="coupon-text">恭喜，您已成功领取<span class="red">满' + CURRENCY_SIGN + data.over_price + '减' + CURRENCY_SIGN + data.ticket_price + '</span>优惠券。</div><div class="coupon-time">使用时间：' + data.start_time + '-' + data.end_time + '</div>';
                            idom = '<i class="icon success-icon"></i>';

                            if (payJiFen) {
                                jiFenObj.attr("data-payjifen", payJiFenCount - payJiFen);
                            }
                            if (interactionJiFen) {
                                jiFenObj.attr("data-interactionjifen", interactionJiFenCount - interactionJiFen);
                            }

                            break;
                        case 1:
                            pdom = '<div class="coupon-text">您所属的会员不能领取</div>';
                            break;
                        case 2:
                            pdom = '<div class="coupon-text">已全部领取</div>';
                            break;
                        case 3:
                            pdom = '<div class="coupon-text">您已经领取</div>';
                            break;
                    }
                    htmls = '<div class="get-coupon-alert"><div class="coupon-alert-inner">' + idom + pdom + '<div class="coupon-href"><a href="/dom/Coupon/Coupon.php?username=' + user_name + '&channel_id=' + data.couponChId + '"> 查看更多优惠券</a><i></i><a href="/dom/my_coupon.php?username=' + user_name + '&ls_cur=8">我的优惠券</a></div><div class="coupon-btn"><a class="enter-btn" href="javascript:;">确定</a></div></div></div>';
                    var winNumber = $.popup({
                        cName: 'evPopupWhite',
                        head: {yes: 0},
                        type: 4,
                        con: {
                            html: htmls
                        }
                    });
                    $('#evPopup_' + winNumber).on('click', '.enter-btn', function () {
                        $.popupClose(winNumber);
                    });
                });
            }

            if (payJiFen || interactionJiFen) {
                var msg = '领取优惠卷,需消耗';
                if (payJiFen) {
                    msg += payJifFenName +'<br />'+ payJiFen + '积分,';
                }
                if (interactionJiFen) {
                    msg += interactionJifFenName + interactionJiFen + '积分,';
                }
                msg += '确定领取吗?';

                $.popup({
                    type: 2,
                    animate: {
                        type: 1
                    },
                    cName: "evPopupWhite",
                    area: {
                        w: 300
                    },
                    con: {
                        text: [1, msg],
                        img: [1, 'warn']
                    },
                    but: {
                        yes: 1,
                        button: {
                            but_1: {
                                text: $weisiteLa.QuXiao,
                                fun: function () {

                                }
                            },
                            but_2: {
                                text: $weisiteLa.QueDing,
                                fun: yes
                            }
                        }
                    }
                });
            } else {
                yes();
            }
        });
        pCouponList.on({
            'mouseenter': function () {
                var $this = $(this),
                    classifyModule = null;
                if ($this.find('.small-classify-wrap-more').length) {
                    classifyModule = $this.find('.classify-module');
                    classifyModule.addClass('classify-module-show');
                    $this.data('show', 1);
                }
            },
            'mouseleave': function () {
                var $this = $(this);
                if ($this.data('show')) {
                    $this.find('.classify-module').removeClass('classify-module-show');
                }
            }
        });
    }
};
// 产品最终页中的联系方式相关的js
wsf.pContactList = function(){
    var pContactList = $('#pContactList');
    if(pContactList.length){
        pContactList.on('click', '.wechat-classify-wrap', function(){
            var $this = $(this),
                dataImg = $this.attr('data-img'),
                htmls = '<div class="show-contact-wechat"><div class="wechat-pic"><img src="' + dataImg + '" /></div><p>'+$weisiteLa.WeiXinSaoYiSao+'</p></div>';
            $.popup({
                cName:'evPopupWhite',
                head : {yes: 0},
                type : 4,
                con : {
                    html : htmls
                }
            });

        });
    }
};

// 产品最终页中的联系方式相关的js
wsf.productPrevNextPosition = function(){
    var productPrevBtn = $('#productPrevBtn'),
        productNextBtn = $('#productPrevBtn');
    if(productPrevBtn.length || productNextBtn.length){
        var bodyW = $('body').width(),
            headH = $('#main_container').offset().top();
        userSiteWidth;
        productPrevBtn.css({

        })
    }
};
/*
    createDate:2017/09/24
    预售js方法（人数列表显示更多）
*/
wsf.presellPeopleListFun = function () {
    var presellPeopleList = $("#presellPeopleList");
    presellPeopleList.on('click', 'small', function () {
        var $this = $(this);
        if (!$this.hasClass('switch-on')) {
            $this.addClass('switch-on');
            presellPeopleList.find('.classify-wrap-hide').show();
        } else {
            $this.removeClass('switch-on');
            presellPeopleList.find('.classify-wrap-hide').hide();
        }
    });
};
/*
    createDate:2017/09/24
    倒计时函数
*/
wsf.countDown = function (j) {
    j.type = j.type || 1;
    var r = function (t) {
            var a = t.split(' '),
                ymd = a[0],
                hms = a[1],
                str = ymd.split('-'),
                fix = hms.split(':'),
                year = str[0] - 0,
                month = str[1] - 0 - 1,
                day = str[2] - 0,
                hour = fix[0] - 0,
                minute = fix[1] - 0,
                second = fix[2] - 0,
                time = (new Date(year, month, day, hour, minute, second)).getTime();
            return parseInt(time / 1000);
        },
        o = j.o,
        st = r(j.st),
        et = r(j.et),
        nts = j.nt ? r(j.nt) : (new Date().getTime() / 1000),
        underway = function (sType) {
            var d, h, mi, s, now = nts,
                c = (sType === 'b' ? st : et) - now,
                html_;
            nts = nts + 1;
            if (c > 0) {
                d = Math.floor(c / (60 * 60 * 24));
                h = Math.floor((c - d * 24 * 60 * 60) / 3600);
                mi = Math.floor((c - d * 24 * 60 * 60 - h * 3600) / 60);
                s = Math.floor(c - d * 24 * 60 * 60 - h * 3600 - mi * 60);
                h = h < 10 ? '0' + h : h;
                mi = mi < 10 ? '0' + mi : mi;
                s = s < 10 ? '0' + s : s;
                switch(j.type){
                    case 1:
                        html_ = '<span class="count-time"><i>' + d + '</i><em>天</em><i>' + h + '</i><em>时</em><i>' + mi + '</i><em>分</em><i>' + s + '</i><em>秒</em></span>';
                        break;
                    case 2:
                        html_ = '<span class="count-time"><i>' + h + '</i><em>:</em><i>' + mi + '</i><em>:</em><i>' + s + '</i></span>';
                        break;
                }
                o.html(html_);
                setTimeout(function () {
                    underway(sType);
                }, 1000);
            } else {
                if (sType === 'b') {
                    underway('n');
                    typeof j.nfun === 'function' && j.nfun();
                } else {
                    typeof j.efun === 'function' && j.efun();
                }
            }
        };
    // 判断状态
    if ((st - nts) > 0) {
        typeof j.sfun == 'function' && j.sfun();
        underway('b')
    } else if ((nts - et) > 0) {
        typeof j.efun == 'function' && j.efun();
    } else {
        underway('n');
        typeof j.nfun == 'function' && j.nfun();
    }
};
/*全屏行方法*/
wsf.fullRow = {
    fullRowXY: null,
    afterRender: null,
    win: window,
    curFullRow: null,
    evRFSwidthmodule: null,
    headFullXY: null,
    scrollbars: null,
    loadFun: function(){
        var that = this,
            btnliItem = null,
            section = '#add_container>.customModuleRow,#add_container>.customModuleRowGroup',
            footPublicModuleRow,
            footer;
        if(that.evRFSwidthmodule.length){
            btnliItem = that.evRFSwidthmodule.find('.btnli-item');
            that.evRFSwidthmodule.on({
                "click": function(){
                    var $this = $(this),
                        tagRow = $this.attr('data-target-row');
                    if(!$this.hasClass('active')){
                        $.scrollify.move('#' + tagRow);
                    }
                }
            }, '.btnli-item');
        }
        if(that.headFullXY.height() > 10){
            section += ', #headFullXY';
        }
        if((footPublicModuleRow = $('.footPublicModuleRow')).length && (footPublicModuleRow.height() > 0)){
            section += ', .footPublicModuleRow';
        }
        /* if((footer = $('#footer')).length && (footer.height() > 0)){
            section += ', #footer';
        } */
        if(that.scrollbars === null){
            that.scrollbars = false;
        }
        $.scrollify({
            section : section,
            sectionName : "full-name",
            // interstitialSection: '#header',
            scrollSpeed: 1100,
            offset : 0,
            scrollbars: false,
            setHeights: false,
            overflowScroll: true,
            updateHash: true,
            touchScroll:true,
            before:function(x, rowArr) {
                btnliItem && btnliItem.each(function(i, dom){
                    dom = $(dom);
                    if(dom.attr('data-target-row') === rowArr[x].attr('data-full-name')){
                        dom.addClass('active');
                    }else{
                        dom.removeClass('active');
                    }
                });
            },
            after:function(x, rowArr) {
                /* var curRow = $(rowArr[x]);
                if(curRow.data('cscroll')){
                    switch(curRow[0].mcs.topPct*1){
                        case 0:
                            curRow.mCustomScrollbar('scrollTo', {
                                y: "2"
                            });
                            break;
                        case 100:
                            curRow.mCustomScrollbar('scrollTo', {
                                y: Math.abs(curRow[0].mcs.topPct) - 2 + ''
                            });
                            break;
                    }
                }
                curRow.find('.load-animate').each(function(i, dom){
                    wsf.f.animationLoadMo($(dom));
                }) */
            },
            afterResize:function() {},
            afterRender:function() {
                that.afterRender = true;
                if(that.evRFSwidthmodule.length){
                    that.evRFSwidthmodule.find('.btnli-item:first').addClass('active');
                }
            }
        });
    },
    addBtnliItem: function(){
        var btnliItemArr = [];
        $("#headFullXY, #add_container>.customModuleRow,#add_container>.customModuleRowGroup,.footPublicModuleRow").each(function(i, dom){
            dom = $(dom);
            dom.attr('data-full-name', dom.attr('id')).data('fullName', dom.attr('id'));
            var text = '通栏';
            switch(dom.attr('id')){
                case 'headFullXY':
                    if(dom.height() < 1){
                        return true;
                    }
                    text = '头部'
                    break;
                case 'footer':
                    text = '底部'
                    break;
                default:
                    text = '通栏'+ i;
                    break;
            }
            if(!dom.closest('.rowGroup-box-inner').length){
                btnliItemArr.push('<li class="btnli-item" data-target-row="' + dom.attr('id') + '"><span class="btnli-item-inner"><b class="btnli-item-text">' + text + '</b><em class="btnli-item-bg"></em></span></li>')
            }
        });
        return btnliItemArr;
    },
    countMoPosition: function () {
        var that = this;
        that.win != window.top && (that.win = window.top);
        that.win = $(that.win);
        if(!that.fullRowXY){
            $('div.customModuleRowFullXY').each(function(i, dom){
                dom = $(dom);
                !that.fullRowXY && (that.fullRowXY = {});
                that.fullRowXY[dom.attr('id')] = {
                    rowDom: dom,
                    originH: dom.find('.CModulePA').height()
                }
            });
            that.headFullXY = $('#headFullXY');
            that.evRFSwidthmodule = that.headFullXY.find('.ev-RFSwidth-module');
            if(that.evRFSwidthmodule.length){
                that.evRFSwidthmodule = that.evRFSwidthmodule.closest('.ev-module-edit');
                !is_action && that.evRFSwidthmodule.css('position','fixed');
                var btnliItemArr = that.addBtnliItem();
                btnliItemArr.length && that.evRFSwidthmodule.find('ul').html(btnliItemArr.join(''));
            }
        }
        var winH = Math.max(that.win[0].innerHeight, 500);
        if (that.fullRowXY !== null) {
            var k_ = 0;
            $.each(that.fullRowXY, function (k, val) {
                var rowOriginHeight = val.originH * 1;
                val['offsetTop'] = val.rowDom.offset().top;
                if (rowOriginHeight < winH) {
                    val.rowDom.find('.CModulePA').height(winH).children('.ev-module-edit, .customModule').each(function (i, moDom) {
                        moDom = $(moDom);
                        var moDomH = moDom.height();
                        if(!moDom.closest('.ev-container-wap').length && !moDom.data('sizeH')){
                            /* var moOriginTop = moDom.position().top,
                                newTop = (moOriginTop / (rowOriginHeight - moDomH)) * (winH - moDomH);
                            moDom.css('top', newTop + 'px'); */
                            moDom.css('top', (moDom.position().top / rowOriginHeight)*100+'%');
                            moDom.data('sizeH', 1);
                        }
                    });
                }
                if (k_ === 0 && rowOriginHeight > (winH + 10)){
                    // that.scrollbars != null  && $.scrollify.setOptions({scrollbars: true});
                    that.scrollbars = true;
                }
                k_++;
                // 这是添加行的滚动条，暂时不用了
                /*if (rowOriginHeight > (winH + 10)) {
                    if(!is_action){
                        val.rowDom.css({
                            'overflow': 'hidden',
                            'height': winH + 'px'
                        });
                        wsf.f.mCScrollbar(val.rowDom);
                    }
                } */
            });
            // 这是添加顶部的滚动条，暂时不用了
            /* if(!is_action){
                if(that.headFullXY.length && (that.headFullXY.height() > (winH + 10))){
                    that.headFullXY.css({
                        'overflow': 'hidden',
                        'height': winH + 'px'
                    });
                    wsf.f.mCScrollbar(that.headFullXY);
                }
            } */
            !is_action && !that.afterRender && that.loadFun();
        }else{
            !is_action && that.evRFSwidthmodule.remove();
        }
        if(that.evRFSwidthmodule.length && !is_action){
            if(that.evRFSwidthmodule.attr('data-position') * 1){
                wsf.f.positionShow(that.evRFSwidthmodule, that.evRFSwidthmodule.attr('data-position') - 1);
            }else{
                that.evRFSwidthmodule.css({
                    left: Math.floor(that.evRFSwidthmodule.attr('data-lshifting') * 1 + ($('body').width() - userSiteWidth) / 2) + 'px',
                    top: that.evRFSwidthmodule.attr('data-tshifting') + 'px'
                });
            }
        }
    }
};
/*
    createDate:2017/09/24
    旧手动左右滚动产品效果――类
*/
wsf.flow_pro_ = function (n, t) {
    var ContN = $(n).parent(),
        $list_ul = ContN.find("ul"),
        $list_ul_par = $list_ul.parent(),
        $list_li_num = $list_ul.find("li").length,
        $list_li_Lborder = parseInt($list_ul.find("li").css("border-left-width")),
        $list_li_Rborder = parseInt($list_ul.find("li").css("border-right-width")),
        $list_li_Lpadd = parseInt($list_ul.find("li").css("padding-left")),
        $list_li_Rpadd = parseInt($list_ul.find("li").css("padding-left")),
        $list_li_Rmargin = parseInt($list_ul.find("li").css('margin-right')),
        $list_li_Lmargin = parseInt($list_ul.find("li").css('margin-left'));
    $list_li_Lborder = isNaN($list_li_Lborder) ? 0 : $list_li_Lborder;
    $list_li_Rborder = isNaN($list_li_Rborder) ? 0 : $list_li_Rborder;
    $list_li_Rmargin = isNaN($list_li_Rmargin) ? 0 : $list_li_Rmargin;
    $list_li_Lmargin = isNaN($list_li_Lmargin) ? 0 : $list_li_Lmargin;
    var $list_li_width = $list_ul.find("li").width() + $list_li_Lborder + $list_li_Rborder + $list_li_Lmargin + $list_li_Rmargin + $list_li_Lpadd + $list_li_Rpadd,
        $show_num = Math.floor(($list_ul_par.width()) / $list_li_width),
        $ul_width = $list_li_width * $list_li_num,
        $list_ul_left = Math.abs($list_ul.position().left),
        $page = Math.ceil($list_li_num / $show_num),
        $move = $show_num * $list_li_width,
        $newMove = $ul_width - ($list_ul_left + $move);
    if (t == 1) {
        if ($newMove > 0) {
            if ($newMove > $move) {
                if ($list_ul.is(':animated') === false) {
                    $list_ul.animate({
                        "left": "-=" + $move
                    }, 500);
                    $(n).parent().find(".left-but").addClass("left-but-off");
                }
            } else {
                if ($list_ul.is(':animated') === false) {
                    $list_ul.animate({
                        "left": "-=" + $newMove
                    }, 500);
                    $(n).parent().find(".right-but").addClass("right-but-off");
                    $(n).parent().find(".left-but").addClass("left-but-off");
                }
            }
        }
    } else {
        if ($list_ul_left > 0) {
            if ($list_ul_left > $move) {
                if ($list_ul.is(':animated') === false) {
                    $list_ul.stop().animate({
                        "left": "+=" + $move
                    }, 500);
                    $(n).parent().find(".right-but").removeClass("right-but-off");
                }
            } else {
                if ($list_ul.is(':animated') === false) {
                    $list_ul.stop().animate({
                        "left": "+=" + $list_ul_left
                    }, 500);
                    $(n).parent().find(".left-but").removeClass("left-but-off");
                    $(n).parent().find(".right-but").removeClass("right-but-off");
                }
            }
        }
    }
};
wsf.flow_pro = function () {
    $(".right-but").click(function () {
        wsf.flow_pro_(this, 1);
    });
    $(".left-but").click(function () {
        wsf.flow_pro_(this, 2);
    });
};
/*
    createDate:2017/09/24
    在线客服
*/
wsf.onlineService = function () {
    var onService_panel = $("#onService_panel");
    onService_panel.on({
        mouseenter: function () {
            onService_panel.animate({
                right: 0
            });
            $(this).hide();
        }
    }, "#onlineOpen");
    onService_panel.on({
        click: function () {
            onService_panel.animate({
                right: -102
            });
            onService_panel.find(".online_tab").fadeOut(100);
            onService_panel.find("#onlineOpen").show();
        }
    }, "#onlineClose");

    $(".online_icon").click(function () {
        $(".online_tab").fadeOut(100);
        var onclickId = $(this).attr("id");
        var findparent_tab;
        switch (onclickId) {
            case "online_tel_icon":
                findparent_tab = $("#onlineTelTab");
                break;
            case "online_qq_icon":
                findparent_tab = $("#onlineQQTab");
                break;
            case "online_message_icon":
                findparent_tab = $("#onlineMessageTab");
                break;
        }
        findparent_tab.fadeIn(100);
    });
    $("#onService_panel .tab_close").click(function () {
        $(this).parents(".online_tab").hide();
    });

    function checkLen(obj, maxs) {
        var maxChars = maxs; //最多字符数
        if (obj.value.length > maxChars) {
            obj.value = obj.value.substring(0, maxChars);
        }
        var curr = maxChars - obj.value.length;
        $(obj).parents("dl").find(".text_length b").text(curr.toString());
    }
};
/**
 * 新版在线客服
 */

wsf.customerService = {
    customerServiceDom: null,
    changeProps: function(){
        var customerService = window.GData.customerService;
        this.customerServiceDom.attr({'data-style': customerService.style, 'data-position-type': customerService.positionType, 'class':'customer-service service-style-' + customerService.style}).data({'style': customerService.style, 'position-type': customerService.positionType});
    },
    renderList: function(){
        var customerService = null;
        var sDom = null;
        var htmlArr = [];
        var customerServiceIsNew = null;

        if(window.GData){
            customerServiceIsNew = window.GData.customerServiceIsNew;
            customerService = window.GData.customerService
        }
        if(!customerServiceIsNew){
            return;
        }else{
            if(!this.customerServiceDom){
                this.customerServiceDom = $('<div class="customer-service" id="customerService" data-change="0"></div>');
                wsf.pDom.wrapper.append(this.customerServiceDom);
            }
        }
        sDom = this.customerServiceDom;
        if(!sDom || !sDom.length){
            return;
        }
        if(customerService){
            this.changeProps();
        }
        if(customerService && customerService.show){
            htmlArr.push('<div class="item-list">');
            customerService.itemList.forEach(function(item, index){
                if(item.show){
                    // 组装项目
                    htmlArr.push('<div class="item-box '+ item.id +'-item">');
                    
                    if(item.p_href){
                        htmlArr.push('<a class="item-href" href="'+ (item.p_href||'javascript:viod(0)') +'" target="'+(item.p_target ==0 ?'self':'_blank')+'"></a>');
                    }
                    // 组装图标
                    htmlArr.push('<div class="icon-box">\
                                    <span class="evIcon '+ item.icon +'"></span>\
                                </div>\
                                ');
                    // 组装标题
                    htmlArr.push('<div class="text-box">\
                                    <span class="text">'+ item.title +'</span>\
                                </div>\
                                ');
                    // 组装更多项
                    if(item.subList && item.subList.length){
                        var itemArr = [];
                        var moreClass = '';
                        switch(item.id){
                            case 'wechat':
                                if(item.subList.length > 1){
                                    moreClass = ' wechat-style-2';
                                }else{
                                    moreClass = ' wechat-style-1';
                                }
                                break;
                        }
                        item.subList.forEach(function(subItem, subIndex){
                            var itemHTML = '';
                            switch(item.id){
                                case 'tel':
                                    itemHTML = '<a class="more-item" href="tel:'+ subItem.value +'">' +
                                                    '<span class="evIcon'+(subItem.pic?'':' evIcon-phone-5')+'">'+ (subItem.pic ? '<img src="'+ subItem.pic +'"/>': '') +'</span>' + 
                                                    '<div class="more-item-attr">\
                                                        <strong class="tit">'+ subItem.title +'</strong>\
                                                        <span class="val">'+ subItem.value +'</span>\
                                                    </div>\
                                                </a>';
                                    break;
                                case 'online':
                                    (function(){
                                        var href = '';
                                        var icon = '';
                                        switch(subItem.type*1){
                                            case 1:
                                                href = 'tencent://message/?uin='+ subItem.value +'&site=qq&menu=yes';
                                                icon = 'http://aimg8.dlszywz.com/tj/customerService/service_qq.png'
                                                break;
                                            case 2:
                                                href = 'https://amos.alicdn.com/msg.aw?v=2&uid='+ subItem.value +'&site=cnalichn&s=11&charset=utf-8';
                                                icon = 'http://aimg8.dlszywz.com/tj/customerService/service_aliTM.png'
                                                break;
                                            case 3:
                                                href = 'skype:'+ subItem.value +'?call';
                                                icon = 'http://aimg8.dlszywz.com/tj/customerService/service_skype.png'
                                                break;
                                            case 4:
                                                href = 'https://wa.me/'+ subItem.value;
                                                icon = 'http://aimg8.dlszywz.com/tj/customerService/service_whatsapp.png'
                                                break;
                                        }
                                        itemHTML = '<a class="qq-item" href="'+ href +'" target="_blank">\
                                                        <span class="evIcon">\
                                                            <img src="'+(subItem.pic || icon)+'" />\
                                                        </span>\
                                                        <strong class="tit">'+ subItem.title +'</strong>\
                                                    </a>';
                                    })();
                                    break;
                                case 'email':
                                    itemHTML = '<a class="more-item" href="mailto:'+ subItem.value +'">'+
                                                    '<span class="evIcon'+(subItem.pic?'':' evIcon-email-8')+'">'+ (subItem.pic ? '<img src="'+ subItem.pic +'"/>': '') +'</span>' + 
                                                    '<div class="more-item-attr">\
                                                        <strong class="tit">'+ subItem.title +'</strong>\
                                                        <span class="val">'+ subItem.value +'</span>\
                                                    </div>\
                                                </a>'
                                    break;
                                case 'wechat':
                                    itemHTML = '<div class="er-code-box">\
                                                    <div class="code-pic">\
                                                        <img src="'+ subItem.pic +'" />\
                                                    </div>\
                                                    <div class="code-text">'+ subItem.title +'</div>\
                                                </div>';
                                    break;
                            }
                            
                            itemArr.push(itemHTML);
                        });
                        htmlArr.push('\
                            <div class="item-more">\
                                <div class="more-box more-box-'+ item.id + moreClass +'">\
                                    <div class="more-box-inner">\
                                    '+ itemArr.join('')+'\
                                    </div>\
                                </div>\
                            </div>\
                        ');
                    }
                    htmlArr.push('</div>');
                }
            });
            // 组装底部按钮
            // htmlArr.push('<div class="item-box to-top-item" title="回到顶部"><div class="icon-box"><div class="evIcon evIcon-to-up-1"></div></div><div class="text-box"><span class="text">回到顶部</span></div></div>');
            htmlArr.push('</div>');
            
            sDom.html(htmlArr.join(''));
            sDom.on('click', '.item-box', function(){
                if($(this).hasClass('toTop-item')){
                    window.scrollTo(0, 0);
                }
            });
        }
    },
    showToTop: function(st, h){
        if(this.customerServiceDom){
            var toTopItem = this.customerServiceDom.find('.toTop-item');
            if(toTopItem.length){
                if(st > 1 && !toTopItem.hasClass('animate')){
                    toTopItem.addClass('animate');
                    setTimeout(function(){
                        toTopItem.addClass('show');
                    }, 1);
                }else if(st < 1){
                    toTopItem.removeClass('show');
                    setTimeout(function(){
                        toTopItem.removeClass('animate');
                    }, 100);
                }
            }
        }
    }
}

wsf.ev_kf_zc_1 = function () {
    var ev_kf_zc_1 = $('#ev_kf_zc_1');
    ev_kf_zc_1.length && (ev_kf_zc_1.css('margin-top', -ev_kf_zc_1.height() / 2 + 'px'), ev_kf_zc_1.on('click', '.ev_kf_zc_close', function () {
        ev_kf_zc_1.hide();
    }));
};
/*网站宽度加载*/
wsf.loadWidth = function () {
    var windowWidth = wsf.pDom.winDom.width();
    if (windowWidth < userSiteWidth) {
        if(userSiteWidth >= 1920 && windowWidth > 1900){
            wsf.pDom.htmlDom.css({'overflow-x':'hidden'});
        }
        wsf.pDom.bodyDom.width(userSiteWidth);
        wsf.pDom.htmlDom.width(userSiteWidth);
    } else {
        wsf.pDom.bodyDom.width('auto');
        wsf.pDom.htmlDom.width('auto');
    }
};
wsf.windowResize = function () {
    wsf.pDom.winDom.on({
        resize: function () {
            wsf.loadWidth();
        }
    });
};
wsf.moduleBind = function () {
    $("div.customModule").on({
        mouseenter: function () {
            var t = $(this),
                dataType = wsf.f.s_j(t.attr('data-attr')),
                childDiv = t.find('.MoBodyC > div');
            if (!childDiv.data('bindevent') && !childDiv.hasClass('editMoConBut')) {
                switch (dataType.mt) {
                    case 5:
                    case 6:
                    case 1:
                    case 2:
                        wsf.textListModule(childDiv);
                        break;
                    case 19:
                        wsf.catalogList(childDiv);
                        break;
                    case 21:
                        wsf.menuCatalogModule(childDiv);
                        break;
                }
            }
        }
    });
};

wsf.queryMJsEffect = function (jsData) {
    for (var MId in jsData) {
        var showStyle = parseInt(jsData[MId].showStyle);
        switch (showStyle) {
            case 13:
                (function(MId){
                    setTimeout(function(){
                        wsf.chartInit(jsData[MId].type, MId, jsData[MId].start, jsData[MId].end, jsData[MId].step, jsData[MId].data);
                    }, 200);
                })(MId);
                break;
            case 12:
                (function(MId){
                    setTimeout(function(){
                        wsf.baiduInit(MId, jsData[MId].x, jsData[MId].y, jsData[MId].content, jsData[MId].level);
                    }, 200);
                })(MId);
                break;
        }
    }
};
wsf.tj_zhuce = function () {
    window.location.href = "/dom/zhuce.php?username=" + user_name;
};
/*
    createDate:2017/09/24
    图册模块
*/
wsf.albums = function (evAlbumsModule) {
    var evAlbumsModule = evAlbumsModule || $('div.ev-albums-module');
    if (evAlbumsModule.length) {
        var albumsCollect = function (obj, allObj) {
                if (!(obj.attr('data-click') * 1)) {
                    allObj.attr('data-click', 1);
                    var t = obj.parent(),
                        number = t.find('.number'),
                        numberTxt = number.text() * 1,
                        timestamp = Date.parse(new Date()),
                        url = '/dom/user_collect_add.php?timestamp=' + timestamp,
                        is_detail = t.data("detail"),
                        data = {
                            'title': t.data("name"),
                            'type': t.data("type"),
                            'doc_id': t.data("id"),
                            'ch_id': t.data("chid"),
                            'href': t.data("href"),
                            'username': user_name,
                            'wap': 0
                        };
                    $.ajax({
                        'url': url,
                        type: "POST",
                        async: false,
                        cache: false,
                        data: data,
                        success: function (data) {
                            allObj.attr('data-click', 0);

                            data = parseInt(data);
                            if (data === 1) {
                                allObj.each(function () {
                                    $(this).hide().siblings('.evIcon').show();

                                    var tmpT = $(this).parent(),
                                        tmpNumber = tmpT.find('.number');

                                    numberTxt === 0 && tmpNumber.show();
                                    tmpNumber.text(numberTxt + 1);
                                });
                                wsf.f.alertWindow($weisiteLa.ShouCangChengGongGeRenZhongXinKan, 2);
                            } else if (data === 2) {
                                wsf.f.confirmWindow($weisiteLa.QingXianDengLuHouShouCang, 'warn', function () {
                                    window.location.href = "/dom/denglu.php?username=" + user_name;
                                });
                                return false;
                            } else if (data === 3) {
                                allObj.each(function () {
                                    $(this).hide().siblings('.evIcon').show();

                                    var tmpT = $(this).parent(),
                                        tmpNumber = tmpT.find('.number');

                                    tmpNumber.text(numberTxt - 1);
                                    numberTxt === 1 && tmpNumber.hide();
                                });

                                return false;
                            } else if (data === 4) {
                                wsf.f.alertWindow($weisiteLa.CanShuCuoWu, 1);
                                return false;
                            }
                        }
                    });
                } else {
                    wsf.f.alertWindow($weisiteLa.ZhengZaiQingQiuQingShaoDeng, 0);
                }
                return false;
            },
            userAlbumLike = function (obj, allObj) {
                if (!(obj.attr('data-click') * 1)) {
                    allObj.attr('data-click', 1);
                    //点赞公用
                    // recordId  文章id
                    // type  ch type
                    // style   属性class
                    // operate   like
                    // part   1
                    // html '{n}'
                    // style, operate, location
                    //没有登录通过cookie验证是否点赞
                    var parentDom = obj.parent(),
                        recordId = parentDom.data("id"),
                        type = parentDom.data("type"),
                        operate = parentDom.data("operate"),
                        zz_userid = readCookie('zz_userid'),
                        number = parentDom.find('.number');
                    if (!zz_userid && operate == 'like') {
                        var userLikeRecord = readCookie('user_like_record');
                        var tmpStr = type + '*' + recordId;
                        //已赞过
                        if (userLikeRecord.indexOf(tmpStr) != -1) {
                            wsf.f.alertWindow($weisiteLa.YiZanGuo, 1);
                            return false;
                        }
                        //每日限制100次
                        var tmpArr = userLikeRecord.split('#');
                        if (tmpArr.length >= 100) {
                            wsf.f.alertWindow($weisiteLa.MeiRiZuiDuoDianZan, 1);
                            return false;
                        }
                    }
                    $.ajax({
                        type: 'POST',
                        dataType: 'json',
                        url: '/Public/UserOperate.php',
                        data: {
                            username: user_name,
                            rid: recordId,
                            opt: operate,
                            type: type,
                            part: 1,
                            is_pc: 1,
                            t: Date.parse(new Date())
                        },
                        success: function (data) {
                            if (data.errorcode > 0) {
                                wsf.f.alertWindow(data.errormsg + '！', 1);
                            } else {
                                allObj.attr('data-click', 0);

                                allObj.each(function () {
                                    var tmpT = $(this).parent(),
                                        tmpNumber = tmpT.find('.number');

                                    data.num == 0 ? tmpNumber.hide() : tmpNumber.show();
                                    tmpNumber.text(data.num);
                                    $(this).hide().siblings('.evIcon').show();
                                    if (operate == 'like') {
                                        tmpT.data('operate', "unLike");
                                    } else if (operate == 'unLike') {
                                        tmpT.data('operate', "like");
                                    }
                                });
                            }
                        }
                    });
                } else {
                    wsf.f.alertWindow($weisiteLa.ZhengZaiQingQiuQingShaoDeng, 0);
                }
            };
        evAlbumsModule.each(function (i, dom) {
            dom = $(dom);
            dom.find('.pic-social').on({
                click: function () {
                    var $this = $(this);
                    if (($this.attr('data-default') * 1) === 1) {
                        wsf.f.alertWindow($weisiteLa.ShiLiTuBuNengCaoZuo, 0);
                    } else {
                        var chType = parseInt($this.parent().parent().attr("data-type"));
                        var id = parseInt($this.parent().parent().attr("data-id"));

                        var allObj = null;

                        if ($this.parent().hasClass('like-btn')) {
                            if (chType === 18) {
                                allObj = $this.hasClass('evIcon-hand-1') ? $(".SMALbumSysButLike" + id) : $(".SMALbumSysButLikeCur" + id);
                            } else {
                                allObj = $this;
                            }

                            userAlbumLike($this, allObj);
                        } else {
                            if (chType === 18) {
                                allObj = $this.hasClass('evIcon-heart-2') ? $(".SMALbumSysButCollect" + id) : $(".SMALbumSysButCollectCur" + id);
                            } else {
                                allObj = $this;
                            }
                            albumsCollect($this, allObj);
                        }
                    }
                    return false;
                }
            }, '.evIcon');
            switch (dom.attr('data-opentype') * 1) {
                case 0:
                    dom.on({
                        click: function () {
                            var $this = $(this);
                            $this.attr('data-target') * 1 === 0 ? window.location.assign($this.attr('data-url')) : window.open($this.attr('data-url'), '_blank');
                        }
                    }, '.div-item');
                    break;
                case 1:
                    dom.find('ul').lightGallery({
                        'galleryId': 'lightgallery_' + i,
                        // 'startClass': 'lg-slide',
                        'mode': 'lg-slide-circular',
                        'backdropDuration': 300,
                        'thumbnail': true,
                        'pager': false,
                        'share': false,
                        'download': false,
                        'loop': true,
                        'dynamic': false,
                        'selector': 'div.div-item',
                        'selectorSort': 'data-id',
                        'exThumbImage': 'data-original-src',
                        'exThumbElement': 'img'
                    });
                    break;
            }
        });
    }
};

/*
    createDate:2018/05/24
    系统调用中的点赞收藏功能
*/
wsf.sysDoc = function (targetDom) {
    var docSysM = targetDom || $('div.ev-sys-doc-module');
    if (docSysM.length) {
        var collect = function (obj, allObj) {
                if (!(obj.attr('data-click') * 1)) {
                    allObj.attr('data-click', 1);
                    var t = obj.parent(),
                        number = t.find('.number'),
                        numberTxt = number.text() * 1,
                        timestamp = Date.parse(new Date()),
                        url = '/dom/user_collect_add.php?timestamp=' + timestamp,
                        is_detail = t.data("detail"),
                        data = {
                            'title': t.data("name"),
                            'type': t.data("type"),
                            'doc_id': t.data("id"),
                            'channel_id': t.data("chid"),
                            'username': user_name,
                            'wap': 0
                        };
                    $.ajax({
                        'url': url,
                        type: "POST",
                        async: false,
                        cache: false,
                        data: data,
                        success: function (data) {
                            allObj.attr('data-click', 0);

                            data = parseInt(data);
                            if (data === 1) {
                                allObj.each(function () {
                                    $(this).hide().siblings('.evIcon').show();

                                    var tmpT = $(this).parent(),
                                        tmpNumber = tmpT.find('.number');

                                    numberTxt === 0 && tmpNumber.show();
                                    tmpNumber.text(numberTxt + 1);
                                });
                                wsf.f.alertWindow($weisiteLa.ShouCangChengGongGeRenZhongXinKan, 2);
                            } else if (data === 2) {
                                wsf.f.confirmWindow($weisiteLa.QingXianDengLuHouShouCang, 'warn', function () {
                                    window.location.href = "/dom/denglu.php?username=" + user_name;
                                });
                                return false;
                            } else if (data === 3) {
                                allObj.each(function () {
                                    $(this).hide().siblings('.evIcon').show();

                                    var tmpT = $(this).parent(),
                                        tmpNumber = tmpT.find('.number');

                                    tmpNumber.text(numberTxt - 1);
                                    numberTxt === 1 && tmpNumber.hide();
                                });

                                return false;
                            } else if (data === 4) {
                                wsf.f.alertWindow($weisiteLa.CanShuCuoWu, 1);
                                return false;
                            }
                        }
                    });
                } else {
                    wsf.f.alertWindow($weisiteLa.ZhengZaiQingQiuQingShaoDeng, 0);
                }
                return false;
            },
            like = function (obj, allObj) {
                if (!(obj.attr('data-click') * 1)) {
                    allObj.attr('data-click', 1);
                    //没有登录通过cookie验证是否点赞
                    var parentDom = obj.parent(),
                        recordId = parentDom.data("id"),
                        type = parentDom.data("type"),
                        operate = parentDom.data("operate"),
                        zz_userid = readCookie('zz_userid'),
                        number = parentDom.find('.number');
                    if (!zz_userid && operate == 'like') {
                        var userLikeRecord = readCookie('user_like_record');
                        var tmpStr = type + '*' + recordId;
                        //已赞过
                        if (userLikeRecord.indexOf(tmpStr) != -1) {
                            wsf.f.alertWindow($weisiteLa.YiZanGuo, 1);
                            return false;
                        }
                        //每日限制100次
                        var tmpArr = userLikeRecord.split('#');
                        if (tmpArr.length >= 100) {
                            wsf.f.alertWindow($weisiteLa.MeiRiZuiDuoDianZan, 1);
                            return false;
                        }
                    }
                    $.ajax({
                        type: 'POST',
                        dataType: 'json',
                        url: '/Public/UserOperate.php',
                        data: {
                            username: user_name,
                            rid: recordId,
                            opt: operate,
                            type: type,
                            part: 1,
                            is_pc: 1,
                            t: Date.parse(new Date())
                        },
                        success: function (data) {
                            if (data.errorcode > 0) {
                                wsf.f.alertWindow(data.errormsg + '！', 1);
                            } else {
                                allObj.attr('data-click', 0);

                                allObj.each(function () {
                                    var tmpT = $(this).parent(),
                                        tmpNumber = tmpT.find('.number');

                                    data.num == 0 ? tmpNumber.hide() : tmpNumber.show();
                                    tmpNumber.text(data.num);
                                    $(this).hide().siblings('.evIcon').show();
                                    if (operate == 'like') {
                                        tmpT.data('operate', "unLike");
                                    } else if (operate == 'unLike') {
                                        tmpT.data('operate', "like");
                                    }
                                });
                            }
                        }
                    });
                } else {
                    wsf.f.alertWindow($weisiteLa.ZhengZaiQingQiuQingShaoDeng, 0);
                }
            };
        docSysM.each(function (i, dom) {
            dom = $(dom);
            if(!dom.data('iconclick')){
                dom.on({
                    click: function () {
                        var $this = $(this),
                            type = $this.data('type');
                        switch(type){
                            case 'share':
                                wsf.share.createShareLayer($this);
                                return false;
                                break;
                            case 'like':case 'collect':
                            if (($this.attr('data-default') * 1) === 1) {
                                wsf.f.alertWindow($weisiteLa.ShiLiTuBuNengCaoZuo, 0);
                            } else {
                                var id = parseInt($this.parent().parent().attr("data-id"));

                                var allObj = null;

                                if ($this.parent().hasClass('int-link')) {
                                    allObj = $this.hasClass('evIcon-hand-1') ? $(".MSysDocButLike" + id) : $(".MSysDocButLikeCur" + id);
                                    like($this, allObj);
                                } else {
                                    allObj = $this.hasClass('evIcon-heart-2') ? $(".MSysDocButCollect" + id) : $(".MSysDocButCollectCur" + id);
                                    collect($this, allObj);
                                }
                            }
                            return false;
                            break;
                        }
                    }
                }, '.pic-interact .evIcon');
                dom.data('iconclick',1);
            }
        });
    }
};
/*
    createDate:2017/09/24
    图册模块中点赞收藏功能
*/
wsf.albumsDataLoad = function (type) {
    if (!GData.SMAlbumIds && !GData.SMAlbumSysIds && !GData.docSysIds) {
        return false;
    }

    type = parseInt(type);

    $.post("/Ajax/TJIndex.php?username=" + user_name, {
        type     : 7,
        ids      : GData.SMAlbumIds,
        sysIds   : GData.SMAlbumSysIds,
        docSysIds: GData.docSysIds
    }, function (data) {
        data.success * 1 && ((data.list && data.list.length) || (data.sysList && data.sysList.length)  || (data.docSysList && data.docSysList.length)) && (function () {
            var list = data.list;
            $.each(list, function (i, v) {
                var curSocial = $('#SMALbum' + v.id);
                if (curSocial.length) {
                    var collectCnt = v.collectCnt * 1, likeCnt = v.likeCnt * 1;
                    collectCnt !== 0 && curSocial.find('.collect-btn .number').text(collectCnt);
                    likeCnt !== 0 && curSocial.find('.like-btn .number').text(likeCnt);
                    curSocial.find('.collect-btn .evIcon[data-action="' + (v.isCollect * 1 ? 'del' : 'add') + '"]').show().siblings('.evIcon').hide();
                    curSocial.find('.like-btn .evIcon[data-action="' + (v.isLike * 1 ? 'del' : 'add') + '"]').show().siblings('.evIcon').hide();
                    curSocial.find('.like-btn').attr('data-operate', (v.isLike * 1 ? 'unLike' : 'like'));
                    curSocial.find('.evIcon').attr('data-click', 0);
                }
            });

            var sysList = data.sysList;
            $.each(sysList, function (i, v) {
                var curSocial = $('.SMALbumSys' + v.id);
                if (curSocial.length) {
                    var collectCnt = v.collectCnt * 1, likeCnt = v.likeCnt * 1;
                    collectCnt !== 0 && curSocial.find('.collect-btn .number').text(collectCnt);
                    likeCnt !== 0 && curSocial.find('.like-btn .number').text(likeCnt);
                    curSocial.find('.collect-btn .evIcon[data-action="' + (v.isCollect * 1 ? 'del' : 'add') + '"]').show().siblings('.evIcon').hide();
                    curSocial.find('.like-btn .evIcon[data-action="' + (v.isLike * 1 ? 'del' : 'add') + '"]').show().siblings('.evIcon').hide();
                    curSocial.find('.like-btn').attr('data-operate', (v.isLike * 1 ? 'unLike' : 'like'));
                    curSocial.find('.evIcon').attr('data-click', 0);
                }
            });

            var docSysList = data.docSysList;
            $.each(docSysList, function (i, v) {
                var curSocial = $('.MDocSys' + v.id);
                if (curSocial.length) {
                    var collectCnt = v.collectCnt * 1, likeCnt = v.likeCnt * 1, readCount = v.readCount * 1, replyCount = v.replyCount * 1;
                    collectCnt !== 0 && curSocial.find('.int-collect .number').text(collectCnt);
                    likeCnt !== 0 && curSocial.find('.int-link .number').text(likeCnt);
                    readCount !== 0 && curSocial.find('.int-read .number').text(readCount);
                    replyCount !== 0 && curSocial.find('.int-comment .number').text(replyCount);
                    curSocial.find('.int-collect .evIcon[data-action="' + (v.isCollect * 1 ? 'del' : 'add') + '"]').show().siblings('.evIcon').hide();
                    curSocial.find('.int-link .evIcon[data-action="' + (v.isLike * 1 ? 'del' : 'add') + '"]').show().siblings('.evIcon').hide();
                    curSocial.find('.int-link').attr('data-operate', (v.isLike * 1 ? 'unLike' : 'like'));
                    curSocial.find('.evIcon').attr('data-click', 0);
                }
            });
        })();
    }, "json");

    return true;
};
/*模块搜索*/
wsf.evSearch = {
    load : function(obj){
        this.fun(obj || $("div.ev-search-module"));
    },
    fun : function(obj){
        obj.each(function(i, dom){
            dom = $(dom);
            var evForm = dom.find('.s-f'),
                keyWord = dom.find('input.key-word');

            dom.on({
                click: function(ev){
                    var targetDom = $(ev.target),
                        $this = $(this),
                        actions = '';

                    switch(true){
                        case targetDom.hasClass('s-w-t'):
                        case targetDom.hasClass('s-w-s'):
                        case targetDom.hasClass('s-w-i'):
                            $this.toggleClass('s-c-active');
                            break;
                        case targetDom.hasClass('w-l-e-t'):
                        case targetDom.hasClass('w-l-e-i'):
                        case targetDom.hasClass('w-l-e'):
                            $this.removeClass('s-c-active');
                            $this.find('.s-w-t').text(targetDom.text());
                            var wle = targetDom.closest('.w-l-e');
                            $this.attr({'data-type': wle.attr('data-type'), 'data-url': wle.attr('data-url')});
                            if (parseInt(wle.attr('data-type')) === 76) {
                                evForm.find('input.key-word').val("");
                            }
                            break;
                        case targetDom.hasClass('s-b-icon'):
                        case targetDom.hasClass('s-b-t'):
                        case targetDom.hasClass('s-b-i'):
                            if(is_action){return false}

                            if(!keyWord.val() || (!(keyWord.attr('data-keywordtype') * 1) && (keyWord.val() == keyWord.attr('data-defaultv')))){
                                wsf.f.alertWindow(dom.attr('data-tishi'), 0);
                                keyWord.trigger('focus');
                                return false;
                            }
                            switch(dom.attr('data-type') * 1){
                                case 2:
                                    evForm.attr('action', dom.attr('data-url'));
                                    evForm.submit();
                                    break;
                                default:
                                    // evForm.attr('action', evForm.find('.s-c').attr('data-url').replace('###', evForm.find('input.key-word').val()));

                                    if (parseInt(evForm.find('.s-c').attr('data-type')) == 76) {
                                        if (parseInt(evForm.find('.s-c').attr('data-blankpageid')) > 0) {
                                            actions =  "/"+user_name +"/bk_"+parseInt(evForm.find('.s-c').attr('data-blankpageid'))+".html";
                                        } else {
                                            wsf.f.alertWindow("暂无找到相关页面！", 0);
                                            keyWord.trigger('focus');
                                            return false;
                                        }
                                    } else {
                                        actions =  evForm.find('.s-c').attr('data-url').replace('###', evForm.find('input.key-word').val());
                                    }

                                    window.location.href = actions;
                                    break;
                            }

                            // evForm.submit();
                            break;
                    }
                }
            },'.s-c, .s-b');
            dom.on('focus blur keydown','input.key-word', function(event){
                var curDom = $(this),
                    curDomVal = curDom.val(),
                    curDomDefaultVal = curDom.attr('data-defaultv');
                switch (event.type) {
                    case 'focusin':
                        !(curDom.attr('data-keywordtype') * 1) && (curDomVal == curDomDefaultVal) && (curDom.val(''), evForm.find('.s-k').removeClass('s-k-place'));
                        (curDom.attr('data-keywordtype') * 1) && evForm.find('.s-k').removeClass('s-k-place');
                        break;
                    case 'focusout':
                        (curDomVal == curDomDefaultVal || !curDomVal) && (curDom.val(curDomDefaultVal), evForm.find('.s-k').addClass('s-k-place'));
                        break;
                    case 'keydown':
                        event.keyCode === 13 && evForm.find('.s-b-i').trigger('click');
                        break;
                }
            });
        });
    }
};

/* 跳转行锚点位置的方法 */
wsf.jumpRowAnchor = function(){
    setTimeout(function() {
        if(window === window.top){
            if(window.location.hash.indexOf('#row_') !== -1){
                var header = $('#header'),
                    top_area = $('#top_area'),
                    win = $(window),
                    top_num = win.scrollTop();
                if(header.length && header.data('fixed')*1 === 1){
                    top_num -= header.height();
                }
                if(top_area.length && top_area.data('fixed')*1 === 1){
                    top_num -= top_area.height();
                }
                win.scrollTop(Math.max(top_num, 0));
            }
        }
    }, 10);
};
// 行组切换效果
wsf.rowGroup = function(obj){
    var rowGroupDom = obj ? obj : $('.customModuleRowGroup');
    rowGroupDom.on('click', '.ctrl-sidebar-area em', function(){
        var $this = $(this),
            showTarget = $this.attr('data-target'),
            rowGroup = null;
        if(!$this.hasClass('cur')){
            $this.addClass('cur').siblings('.cur').removeClass('cur');
            rowArray = $this.closest('.customModuleRowGroup').find('.customModuleRow');
            if(rowArray.length > 1){
                rowArray.each(function(i, dom){
                    dom = $(dom);
                    if(dom.attr('id') === showTarget){
                        dom.addClass('customModuleRowBlock').removeClass('customModuleRowNone');
                    }else{
                        dom.hasClass('customModuleRowBlock') && dom.removeClass('customModuleRowBlock').addClass('customModuleRowNone');
                    }
                })
            }

        }
    });
};
// 背景音乐图标
wsf.webSiteBgMusic = function(){
    var webSiteBgMusic = $('#webSiteBgMusic');
    if(webSiteBgMusic.length){
        var bgMusicBtn = $('<i class="evIcon evIcon-qqyinyue" id="webSiteBgMusicBtn"><i>'),
            audio = webSiteBgMusic[0];
        wsf.pDom.htmlDom.append(bgMusicBtn);
        audio.play();
        if(audio.paused){
            bgMusicBtn.addClass('paused').attr('title',"播放背景音乐");
        }else{
            bgMusicBtn.removeClass('paused').addClass('play').attr('title',"暂停背景音乐");
        }
        /* var audioAutoPlay = setInterval(function(){
            if(audio.paused){
                audio.play();
            }else{
                bgMusicBtn.removeClass('paused').addClass('play').attr('title',"暂停背景音乐");
                clearInterval(audioAutoPlay);
            }
        }, 1000); */
        // playFun();
        bgMusicBtn.on('click', function(){
            // playFun('click');
            if(audio.paused){
                bgMusicBtn.removeClass('paused').addClass('play').attr('title',"暂停背景音乐");
                audio.play();
            }else{
                bgMusicBtn.addClass('paused').attr('title',"播放背景音乐");
                audio.pause();
            }
        })
    }
};
// 音频模块
wsf.audioModule = {
    audioDomArr: [],
    load: function(obj){
        var that = this;
        obj = obj || $(".ev-audio-module");
        obj.each(function(i, dom){
            dom = $(dom);
           if(!that.audioDomArr.find(function(v){
                return v.is(dom);
            })){
                that.fun(dom);
                that.audioDomArr.push(dom);
            }
        })
    },
    fun: function(obj){
        var that = this;
        var audioDom = obj.find('audio');
        var configParame = obj.data();
        var duration = 0;
        // 格式化时间
        var timeRules = function(during){
            var s = Math.floor(during / 1) % 60;
            if(s < 10){
                s = '0'+s;
            }
            during = Math.floor(during / 60);
            var i = during % 60;
            if(i < 10){
                i = '0'+i;
            }
            during = Math.floor(during / 60);
            var h = during % 24;
            if(h < 10){
                h = '0'+h;
            }
            during = Math.floor(during / 24);
            var d = during;
            var getStr = i+':'+s;
            if(h*1){
                getStr = h+':' + getStr;
            }
            return getStr;
        };
        var playProgress = function(c, d){
            return c/d*100;
        }
        // 给音频媒体绑定事件
        audioDom.on({
            canplaythrough: function(){
                
            },
            canplay: function(){
                duration = audioDom[0].duration;
                var currentTime = audioDom[0].currentTime;
                obj.find('.cur-time').text(timeRules(currentTime));
                obj.find('.dur-time').text(timeRules(duration));
                obj.find('.play-bar span').css({width: playProgress(currentTime, duration)+'%'})
            },
            timeupdate: function(){
                var currentTime = audioDom[0].currentTime;
                obj.find('.cur-time').text(timeRules(currentTime));
                obj.find('.play-bar-i').css({width: playProgress(currentTime, duration)+'%'})
            },
            ended: function(){
                obj.find('.playing').removeClass('playing').data('playing', 0);
                audioDom[0].pause();
                audioDom[0].currentTime = 0;
                if(configParame.loopPlay){
                    setTimeout(function(){
                        obj.find('[data-action="playBtn"]').click();
                    }, 10);
                }
            }
        });
        obj.on({
            'click': function(ev){
                var $this = $(this);
                var action = $this.data('action');
                switch(action){
                    case 'playBtn':
                        if($this.data('playing') * 1){
                            $this.removeClass('playing').data('playing', 0);
                            audioDom[0].pause();
                        }else{
                            that.audioDomArr.forEach(function(v){
                                if(!v.is(obj) && v.find('.playing').length){
                                    v.find('.playing').click();
                                }
                            })
                            $this.addClass('playing').data('playing', 1);
                            audioDom[0].play();
                        }
                        break;
                    case 'playBar':
                        var w = $this.width();
                        var l = $this.offset().left;
                        var pageX = ev.pageX;
                        var clickW = pageX - l;
                        var getGoTime = clickW * duration/w;
                        audioDom[0].currentTime = Math.floor(getGoTime);
                        break;
                }
            }
        }, '[data-action]');
        if(configParame.autoPlay){
            // audioDom
            obj.find('[data-action="playBtn"]').click();
        }
    }
};
// sphere 720全景 模块
wsf.sphereModule = {
    lang: {
        zoom: '缩放',
        zoomOut: '缩小',
        zoomIn: '放大',
        zoomRange: '缩放',
        moveUp: '上移',
        moveDown: '下移',
        moveLeft: '左移',
        moveRight: '右移',
        download: '下载',
        fullscreen: '全屏展示',
        menu: '更多',
        close: '关闭',
        twoFingers: 'Use two fingers to navigate',
        ctrlZoom: '使用ctrl+滚动来缩放图像',
        loadError: '资源无法加载',
        gallery: '图片列表',
        videoPlay:'播放/暂停',
        videoVolume:'静音开关',
        videoTime:'视频时长',
        description: '详细介绍'
    },
    sphereDomArr: {},
    load: function(obj){
        var that = this;
        obj = obj || $(".ev-sphere-module");
        obj.each(function(i, dom){
            dom = $(dom);
            var moduleId = dom.closest('.ev-module-edit').attr('id');
            if(!that.sphereDomArr[moduleId]){
                that.fun(dom);
            }
        })
    },
    fun: function(obj){
        var that = this;
        var sphereId = obj.find('.sphere-img-box').attr('id');
        var curModule = obj.closest('.ev-module-edit');
        var objData = obj.data();
        var moduleId = curModule.attr('id');
        var moduleData = curModule.data();
        var getConfig = moduleConfig[moduleId];
        if(!getConfig){
            return;
        }
        var firstGalleryItem = getConfig.galleryList[0];
        if(!firstGalleryItem){
            return;
        }
        var animatedValues = {
            pitch: { start: -Math.PI / 2, end: getConfig.defaultPitch },
            yaw: { start: Math.PI, end: getConfig.defaultYaw },
            zoom: { start: 0, end: getConfig.defaultZoomLvl },
            fisheye: { start: 2, end: getConfig.fisheye*1 ? 2: 0 },
        };
        // 创建配置
        var createdConfig = {
            lang: that.lang,
            container: sphereId,
            touchmoveTwoFingers: true,
            mousewheelCtrlKey: false,
            moveInertia: false,
            minFov: 0,
            maxFov: 90,
            defaultZoomLvl: animatedValues.zoom.end,
            defaultYaw: animatedValues.yaw.end,
            defaultPitch: animatedValues.pitch.end,
            fisheye: animatedValues.fisheye.end,
            panorama: firstGalleryItem.panorama,
            caption: firstGalleryItem.options.caption,
            description: firstGalleryItem.options.description,
            navbar: [],
            plugins: []
        };
        // 自动播放
        if(getConfig.autoRotate > 0 || getConfig.readyAnimation > 0 || getConfig.autoRotateNav>0){
            var autostartDelay = null;
            if((getConfig.readyAnimation > 0 && getConfig.autoRotate < 1) || (getConfig.readyAnimation < 1 && getConfig.autoRotate > 0)){
                autostartDelay = 0;
            }
            createdConfig.plugins.push(
                [PhotoSphereViewer.AutorotatePlugin, {
                    autostartDelay: autostartDelay,
                    autostartOnIdle: false,
                    autorotatePitch: animatedValues.pitch.end,
                    autorotateYaw: animatedValues.yaw.end,
                    autorotateSpeed: '1rpm'
                }]
            )
            if(getConfig.autoRotateNav > 0){
                createdConfig.navbar.push('autorotate');
            }
        }
        // 缩放组件
        if(getConfig.zoom>0){
            createdConfig.navbar.push('zoomOut', 'zoomRange', 'zoomIn');
        }
        // 移动组件
        if(getConfig.move>0){
            createdConfig.navbar.push('moveUp', 'moveDown', 'moveLeft', 'moveRight');
        }
        if(objData.type == 3){
            createdConfig.adapter= [PhotoSphereViewer.EquirectangularVideoAdapter, {
                autoplay: true,
                muted: true,
            }];
            createdConfig.panorama = {source: firstGalleryItem.panorama};
            if(getConfig.videoPlay>0){
                createdConfig.navbar.push('videoPlay');
            }
            if(getConfig.videoVolume>0){
                createdConfig.navbar.push('videoVolume');
            }
            if(getConfig.videoTime>0){
                createdConfig.navbar.push('videoTime');
            }
            createdConfig.plugins.push(
                [PhotoSphereViewer.VideoPlugin]
            );
            // console.log(createdConfig.panorama);
        }
        // // 下载组件
        /* if(getConfig.download>0){
            createdConfig.navbar.push('download');
        } */
        // 介绍组件
        if(firstGalleryItem.options.description){
            createdConfig.navbar.push('description');
        }
        // 画廊组件
        if(objData.type == 2){
            getConfig.galleryList.forEach(function(v){
                v.name = v.options.caption;
            });
            createdConfig.plugins.push(
                [PhotoSphereViewer.GalleryPlugin, {
                    visibleOnLoad: true,
                    hideOnClick: false,
                    // handler: function(){
                    // console.log(666)
                    // }
                }]
            );
            createdConfig.navbar.push('gallery');
        }
        // 标题
        if(firstGalleryItem.options.caption){
            createdConfig.navbar.push('caption');
        }
        // 全屏
        if(getConfig.fullscreen>0){
            createdConfig.navbar.push('fullscreen');
        }
        // console.log(createdConfig);
        // 实例化
        var viewer = new PhotoSphereViewer.Viewer(createdConfig);

        if(objData.type == 2){
            var gallery = viewer.getPlugin(PhotoSphereViewer.GalleryPlugin);
            gallery.setItems(getConfig.galleryList);
        }
        if(getConfig.readyAnimation>0){
            var autorotate = viewer.getPlugin(PhotoSphereViewer.AutorotatePlugin);
            function intro() {
              autorotate.stop();
              new PhotoSphereViewer.utils.Animation({
                  properties: animatedValues,
                  duration: 2500,
                  easing: 'inOutQuad',
                  onTick: function(properties){
                    viewer.setOption('fisheye', properties.fisheye);
                    viewer.rotate({ yaw: properties.yaw, pitch: properties.pitch });
                    viewer.zoom(properties.zoom);
                  },
              }).then(function(){
                autorotate.start();
              });
            }
            viewer.addEventListener('ready', intro, { once: true });
        }
        that.sphereDomArr[moduleId] = viewer;
    }
};
// 百度编辑器中的embed代码替换成video代码
wsf.embedToVideo = function(){
    var editor_content_air = $('.editor_content_air');
    editor_content_air.each(function(i, dom){
        dom = $(dom);
        var embeds = dom.find('embed');
        embeds.each(function(i, embed){
            embed = $(embed);
            var url = embed.attr('src');
            var poster = embed.attr('poster');
            if(url.indexOf('.mp4') !== -1){
                embed.before('<video controls="true" src="'+ embed.attr('src') +'" style="width:'+ embed.attr('width') +'px; height:'+ embed.attr('height') +'px" poster="'+ poster +'"></video>');
                embed.remove();
            }
        });
    });
};
/*页面加载完后要执行加载的函数*/
$(function () {
    //获得当前window对象
    wsf.pDom.winDom = $(window);
    //获得当前html对象
    wsf.pDom.htmlDom = $('html');
    //获得当前body对象
    wsf.pDom.bodyDom = $('body');
    // 获取wrapper对象
    wsf.pDom.wrapper = $('#wrapper');
    //给页面中的wsf事件的a链接去掉打开目标属性target
    wsf.pDom.bodyDom.find('a[href*="javascript:wsf"]').each(function (i, dom) {
        $(dom).removeAttr('target');
    });
    //给页面中的wsf事件的a链接绑定事件
    wsf.pDom.bodyDom.on('click', 'a', function () {
        var $this = $(this),
            href= $this.attr('href');
        if(href && href.indexOf('javascript:wsf') === -1){
            if(href.indexOf('showAlert_') != -1){
                wsf.evAlertRow.showAlert(href.substr(10));
                return false;
            }
            if(href.indexOf('showWXCode_') != -1){
                wsf.f.showWXCode(href.substr(11));
                return false;
            }

        }
    });
    // 静音按钮事件
    wsf.pDom.bodyDom.on('click', '.muted-btn', function(){
        var $this = $(this);
        var $parent = $this.parent();
        var video = null;
        if($parent.hasClass('customModuleRow')){
            video = $this.siblings('.rowVideoBg').find('video');
        }else if($parent.hasClass('video-item')){
            video = $this.siblings('video');
        }
        if($this.hasClass('off')){
            $this.removeClass('off');
            video[0].muted = false;
        }else{
            $this.addClass('off');
            video[0].muted = true;
        }
    });
    // 视频媒体播放按钮事件
    wsf.pDom.bodyDom.on('click', '.play-btn', function(){
        var $this = $(this);
        var $parent = $this.parent();
        var video = null;
        if($parent.hasClass('customModuleRow')){
            video = $this.siblings('.rowVideoBg').find('video');
        }else if($parent.hasClass('video-item')){
            video = $this.siblings('video');
        }
        console.log(video)
        if($this.hasClass('off')){
            $this.removeClass('off evIcon-play-2').addClass('evIcon-pause-8');
            video[0].play();
        }else{
            $this.addClass('off evIcon-play-2').removeClass('evIcon-pause-8');
            video[0].pause();
        }
    });
    //图册焦点图收藏点赞功能方法
    wsf.albumsDataLoad();
    //页面主导航的事件方法
    wsf.nav();
    //页面中搜索模块的方法
    wsf.search();
    //页面模块焦点图方法
    wsf.focusPicModule();
    //页面文字列表模块
    wsf.textListModule();
    //设置滚动模块的事件
    wsf.moveModule();
    //模块树形结构分类事件
    wsf.catalogList();
    //模块抽屉分类事件
    wsf.menuCatalogModule();
    //详情页标签切换方法
    wsf.tabSwitchModule();
    // wsf.tableModule();
    //通栏产品分类的事件
    wsf.classifyModule();
    wsf.selectState();
    wsf.customEditModule();
    wsf.manualMoveModule();
    wsf.innerWapBuy();
    wsf.picTextList_changePic();
    wsf.formModule.defaultValVerify();
    wsf.evTabContent.loadBind();
    wsf.evSearch.load();
    wsf.audioModule.load();
    wsf.videoModule.load();
    wsf.sphereModule.load();
    wsf.banner3D.init();
    // 渲染在线客服列表
    wsf.customerService.renderList();
    if (!is_action) {
        wsf.loadWidth();
        wsf.absoluteMenuCatalog();
        wsf.windowResize();
        wsf.albums();
        wsf.sysDoc();
        wsf.rowVideo();
        wsf.share.bindEvent();
        wsf.concern();
        wsf.pShowGetCoupon();
        wsf.pContactList();
        // wsf.Mp3PlayFun();
        wsf.fixedRow.getDom();
        wsf.scrollFixed.judge();
        wsf.jumpRowAnchor();
        // wsf.fullRow.loadFun();
        // 内页相关产品中的幻灯片展示
        $('.head-body-module').each(function(i,v){
            $(v).find('.pic-list-lightGallery ul').lightGallery({
                'mode': 'lg-slide-circular-vertical',
                'backdropDuration': 300,
                'thumbnail': true,
                'pager': false,
                'share': false,
                'download': false,
                'loop': true,
                'dynamic':false
            });
        });
        wsf.upUUPV();
        wsf.translatorModule.load();
    } else {
        // wsf.moduleBind();
        wsf.absoluteModuleWrapBind();
    }
    // 城市切换
    wsf.cityModule.load();

    wsf.queryMJsEffect(MJsData);
    // 页面加载完执行弹窗方法
    wsf.evAlertRow.loadBind();
    wsf.progress.judge();
    // 页面加载完执行动画加载方法
    wsf.loadAnimate.loadFun();
    wsf.interactFun_();
    wsf.ev_kf_zc_1();
    wsf.wScroll();
    // wsf.rowGroup();
    wsf.webSiteBgMusic();
    // 页面加载完替换百度编辑器中的embed的flash播放器
    wsf.embedToVideo();
    var userAgent = navigator.userAgent;
    if(userAgent.indexOf('iPhone OS') !=-1){
        $('#banner_area .video-item').each(function(i, dom){
            dom = $(dom);
            setTimeout(function(){
                if(dom.find('video')[0].paused){
                    dom.append('<i class="play-btn evIcon evIcon-play-4"></i>');
                }
            }, 1500);
        });
        wsf.pDom.bodyDom.on('click', '.video-item .play-btn', function(){
            $(this).hide().siblings('video')[0].play();
        });
    }
});