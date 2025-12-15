// Game State
var board = null
var game = new Chess()
var playerColor = 'w' // Defaults to White

// jQuery elements
var $status = $('#status')
var $fen = $('#fen')
var $pgn = $('#pgn')
var $eval = $('#evaluation')
var $bar = $('#eval-bar')

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false

    // Only allow player to move their own pieces
    if ((playerColor === 'w' && piece.search(/^b/) !== -1) ||
        (playerColor === 'b' && piece.search(/^w/) !== -1)) {
        return false
    }

    // Check turn
    if ((game.turn() === 'w' && playerColor !== 'w') ||
        (game.turn() === 'b' && playerColor !== 'b')) {
        return false
    }
}

function onDrop(source, target) {
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    })

    if (move === null) return 'snapback'

    updateStatus()
    // Trigger AI move
    window.setTimeout(makeAIMove, 250)
}

function onSnapEnd() {
    board.position(game.fen())
}

function updateStatus() {
    var status = ''
    var moveColor = (game.turn() === 'b') ? 'Black' : 'White'

    if (game.in_checkmate()) {
        status = 'Game over, ' + moveColor + ' is in checkmate.'
    } else if (game.in_draw()) {
        status = 'Game over, drawn position'
    } else {
        status = moveColor + ' to move'
        if (game.in_check()) {
            status += ', ' + moveColor + ' is in check'
        }
    }

    $status.html(status)
    $pgn.html(game.pgn())
}

function makeAIMove() {
    if (game.game_over()) return

    // If it's not AI's turn, do nothing (safety check)
    // AI is color opposite to playerColor
    if (game.turn() === playerColor) return

    $status.text("AI is Thinking (MCTS)...");

    $.ajax({
        url: '/move',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ fen: game.fen() }),
        success: function (response) {
            if (response.move) {
                var from = response.move.substring(0, 2);
                var to = response.move.substring(2, 4);
                var promotion = response.move.length > 4 ? response.move.substring(4, 5) : undefined;

                game.move({ from: from, to: to, promotion: promotion });
                board.position(game.fen());

                var evalVal = response.eval.toFixed(2);
                $eval.text(evalVal);

                var pct = 50 + (response.eval * 50);
                pct = Math.max(0, Math.min(100, pct));
                $bar.css('width', pct + '%');

                updateStatus();
            }
        },
        error: function (err) {
            console.error("AI Error:", err);
            $status.text("AI Error. Check console.");
        }
    });
}

// Emoji Map
function pieceTheme(piece) {
    // Use the SAME solid glyphs for both sides to ensure consistent weight and visibility
    // We will color them using SVG fill/stroke
    var pieceType = piece.charAt(1).toLowerCase();
    var solidMap = {
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    };
    var emoji = solidMap[pieceType] || '';

    var isWhite = piece.charAt(0) === 'w';
    var fillColor = isWhite ? '#f0f0f0' : '#111';
    var strokeColor = isWhite ? '#111' : '#fff';

    var svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
        <!-- Shadow/Outline for visibility -->
        <text x="50%" y="54%" dominant-baseline="central" text-anchor="middle" font-size="60" fill="rgba(0,0,0,0.2)">${emoji}</text>
        <!-- Main Piece -->
        <text x="50%" y="50%" dominant-baseline="central" text-anchor="middle" font-size="60" 
              fill="${fillColor}" 
              stroke="${strokeColor}" 
              stroke-width="1.5"
              style="paint-order: stroke fill;">${emoji}</text>
    </svg>`;
    return 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svg)));
}

// Move Hints Logic
function removeGreySquares() {
    $('#myBoard .square-55d63').css('background', '')
}

function greySquare(square) {
    var $square = $('#myBoard .square-' + square)

    var background = '#a9a9a9'
    if ($square.hasClass('black-3c85d')) {
        background = '#696969'
    }

    // Using CSS class is cleaner but chessboardjs overrides background-color directly often.
    // Let's use the radial gradient approach via a class if possible, or direct style.
    // The user asked for "dots".
    // style.css has .move-hint that does radial gradient.
    // However, chessboard.js squares have specific classes.
    // Let's toggle the class.

    $square.addClass('move-hint')
}

function onMouseoverSquare(square, piece) {
    // get list of possible moves for this square
    var moves = game.moves({
        square: square,
        verbose: true
    })

    // exit if there are no moves available for this square
    if (moves.length === 0) return

    // highlight the square they moused over
    // greySquare(square) 

    // highlight the possible squares for this piece
    for (var i = 0; i < moves.length; i++) {
        greySquare(moves[i].to)
    }
}

function onMouseoutSquare(square, piece) {
    $('#myBoard .square-55d63').removeClass('move-hint')
}

var config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    onMouseoverSquare: onMouseoverSquare,
    onMouseoutSquare: onMouseoutSquare,
    pieceTheme: pieceTheme
}
board = Chessboard('myBoard', config)

updateStatus()

// --- Controls ---

function resetGame(newColor) {
    playerColor = newColor;
    game.reset();
    board.start();
    board.orientation(newColor === 'w' ? 'white' : 'black');
    updateStatus();
    $eval.text("0.0");
    $bar.css('width', '50%');

    // If playing as Black, AI (White) must move first
    if (playerColor === 'b') {
        window.setTimeout(makeAIMove, 500);
    }
}

$('#playWhiteBtn').on('click', function () {
    resetGame('w');
});

$('#playBlackBtn').on('click', function () {
    resetGame('b');
});

$('#undoBtn').on('click', function () {
    // Determine how many moves to undo.
    // If it's my turn, undo 2 (AI + Me)
    // If AI is thinking... problem. But we assume button clicked during user turn.

    // Undo AI move
    game.undo();
    // Undo Player move
    game.undo();

    board.position(game.fen());
    updateStatus();
});

$('#flipBtn').on('click', function () {
    board.flip();
});
