// Why a game ended, in words for the game-over card. Pure: takes the chess.js game and which colour the player has.

import type { Chess } from 'chess.js';

export interface GameOverInfo {
  /** "Checkmate, you win" */
  title: string;
  /** one sentence with the rule behind it */
  reason: string;
  /** 1 = player won, 0 = draw, -1 = player lost */
  result: 1 | 0 | -1;
}

export function gameOverInfo(game: Chess, human: 'w' | 'b'): GameOverInfo | null {
  if (game.isCheckmate()) {
    const loser = game.turn();
    const youWon = loser !== human;
    return {
      title: youWon ? 'Checkmate, you win' : 'Checkmate, the engine wins',
      reason: `${loser === 'w' ? 'White' : 'Black'} is in check and has no legal move.`,
      result: youWon ? 1 : -1,
    };
  }
  if (!game.isGameOver()) return null;
  let reason = 'The game is drawn.';
  if (game.isStalemate()) reason = `Stalemate: ${game.turn() === 'w' ? 'White' : 'Black'} has no legal move but is not in check.`;
  else if (game.isInsufficientMaterial()) reason = 'Neither side has enough pieces left to give checkmate.';
  else if (game.isThreefoldRepetition()) reason = 'The same position has occurred three times.';
  else if (game.isDrawByFiftyMoves()) reason = 'Fifty moves have passed without a capture or a pawn move.';
  return { title: 'Draw', reason, result: 0 };
}
