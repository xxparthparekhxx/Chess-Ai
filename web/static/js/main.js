var board = null
var game = new Chess()
var $status = $('#status')
var $fen = $('#fen')
var $pgn = $('#pgn')
var $eval = $('#evaluation')
var $bar = $('#eval-bar')

function onDragStart(source, piece, position, orientation) {
    // do not pick up pieces if the game is over
    if (game.game_over()) return false

    // only pick up pieces for the side to move
    if (game.turn() === 'w' && piece.search(/^b/) !== -1) return false
    if (game.turn() === 'b' && piece.search(/^w/) !== -1) return false
}

function onDrop(source, target) {
    // see if the move is legal
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q' // NOTE: always promote to a queen for example simplicity
    })

    // illegal move
    if (move === null) return 'snapback'

    updateStatus()

    // Make AI Move
    makeAIMove()
}

// update the board position after the piece snap
// for castling, en passant, pawn promotion
function onSnapEnd() {
    board.position(game.fen())
}

function updateStatus() {
    var status = ''

    var moveColor = 'White'
    if (game.turn() === 'b') {
        moveColor = 'Black'
    }

    // checkmate?
    if (game.in_checkmate()) {
        status = 'Game over, ' + moveColor + ' is in checkmate.'
    }

    // draw?
    else if (game.in_draw()) {
        status = 'Game over, drawn position'
    }

    // game still on
    else {
        status = moveColor + ' to move'

        // check?
        if (game.in_check()) {
            status += ', ' + moveColor + ' is in check'
        }
    }

    $status.html(status)
    $pgn.html(game.pgn())
}

function makeAIMove() {
    if (game.game_over()) return;

    $status.text("AI is Thinking...");

    $.ajax({
        url: '/move',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ fen: game.fen() }),
        success: function (response) {
            if (response.move) {
                var move = game.move(response.move, { loose: true }); // UCI format needs loose parsing sometimes or from/to
                // Chess.js 'move' can take string like 'e2e4' if sloppy? No.
                // We need to parse UCI 'e2e4' to {from: 'e2', to: 'e4'}

                var from = response.move.substring(0, 2);
                var to = response.move.substring(2, 4);
                var promotion = response.move.length > 4 ? response.move.substring(4, 5) : undefined;

                game.move({ from: from, to: to, promotion: promotion });
                board.position(game.fen());

                // Update eval
                var evalVal = response.eval.toFixed(2);
                $eval.text(evalVal);

                // Update bar (sigmoid-ish or clamp?)
                // Simple clamp -1 to 1 -> 0% to 100%
                var pct = 50 + (response.eval * 50);
                pct = Math.max(0, Math.min(100, pct));
                $bar.css('width', pct + '%');

                updateStatus();
            } else {
                console.log(response);
            }
        },
        error: function (err) {
            console.error("AI Error:", err);
            $status.text("AI Crashed :(");
        }
    });
}

// Emoji Map
function pieceTheme(piece) {
    var pieceMap = {
        'wP': '♙', 'wN': '♘', 'wB': '♗', 'wR': '♖', 'wQ': '♕', 'wK': '♔',
        'bP': '♟', 'bN': '♞', 'bB': '♝', 'bR': '♜', 'bQ': '♛', 'bK': '♚'
    };
    var emoji = pieceMap[piece] || '';

    // Create an SVG data URI
    // We center the text and use a large font size
    var svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
        <text x="50%" y="50%" dominant-baseline="central" text-anchor="middle" font-size="60" fill="gray">${emoji}</text>
        <text x="50%" y="50%" dominant-baseline="central" text-anchor="middle" font-size="60" fill="${piece.charAt(0) === 'w' ? '#f0f0f0' : '#1a1a1a'}" stroke="${piece.charAt(0) === 'w' ? '#000' : '#fff'}" stroke-width="2">${emoji}</text>
    </svg>`;

    return 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svg)));
}

var config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    pieceTheme: pieceTheme
}
board = Chessboard('myBoard', config)

updateStatus()

// Controls
$('#resetBtn').on('click', function () {
    game.reset();
    board.start();
    updateStatus();
    $eval.text("0.0");
    $bar.css('width', '50%');
});

$('#flipBtn').on('click', function () {
    board.flip();
});
