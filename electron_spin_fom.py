import numpy as np
from quocslib.utils.AbstractFoM import AbstractFoM
import qutip as qt
from typing import List, Dict, Optional, Any
from scipy.linalg import expm  # SciPyのexpm関数を使用

class ElectronSpinQubit(AbstractFoM):
    """
    GRAPE最適化のための電子スピン量子ビット系のFigure of Merit (FoM) クラス
    3つの制御（X, Y, Z方向）を持ちます
    
    このクラスは量子状態間の忠実度を最大化します
    """
    
    def __init__(self, args_dict: Optional[Dict[str, Any]] = None):
        """
        初期化関数
        
        Args:
            args_dict: 設定パラメータの辞書（オプション）
        """
        if args_dict is None:
            args_dict = {}
        
        # 最適化方向の設定
        self.is_maximization = args_dict.setdefault("is_maximization", True)
        self.FoM_factor = 1
        if not self.is_maximization:
            self.FoM_factor = -1
            
        # 電子スピン量子ビットのパラメータ
        self.initial_state_name = args_dict.setdefault("initial_state", "down")
        self.target_state_name = args_dict.setdefault("target_state", "up")
        self.total_time = args_dict.setdefault("total_time", 10.0)
        self.n_time_steps = args_dict.setdefault("time_steps", 100)
        self.dt = self.total_time / self.n_time_steps
        
        # 電子スピンの演算子をNumPy配列として定義
        # Pauli行列
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        identity = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        
        # スピン状態（密度行列として）
        self.up_dm = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        self.down_dm = np.array([[0, 0], [0, 1]], dtype=np.complex128)
        
        # 制御ハミルトニアン
        self.H_controls = [sx, sy, sz]
        
        # ドリフトハミルトニアン（自由発展項）- 零行列
        self.H_drift = np.zeros((2, 2), dtype=np.complex128)
        
        # 状態の設定（密度行列）
        self.initial_state = self._get_state(self.initial_state_name)
        self.target_state = self._get_state(self.target_state_name)
        
        # プロパゲータの保存用
        self.n_slices = self.n_time_steps
        self.prop_store = [np.zeros((2, 2), dtype=np.complex128) for _ in range(self.n_slices)]
        self.propagators_are_computed = False
        
        # システムタイプの設定（GRAPEアルゴリズム用）
        self.sys_type = "StateTransfer"
    
    def _get_state(self, state_str: str) -> np.ndarray:
        """
        状態名から対応する密度行列を返す
        
        Args:
            state_str: 状態の名前 ("up" または "down")
            
        Returns:
            状態の密度行列（NumPy配列）
        """
        if state_str.lower() == "up":
            return self.up_dm
        elif state_str.lower() == "down":
            return self.down_dm
        else:
            # デフォルトは "down"
            return self.down_dm
    
    def get_control_Hamiltonians(self):
        """制御ハミルトニアンを返す"""
        return self.H_controls
    
    def get_drift_Hamiltonian(self):
        """ドリフトハミルトニアンを返す"""
        return self.H_drift
    
    def get_initial_state(self):
        """初期状態を返す"""
        return self.initial_state
    
    def get_target_state(self):
        """目標状態を返す"""
        return self.target_state

    def get_propagator(self, 
                       pulses_list: List[np.ndarray] = [],
                       time_grids_list: List[np.ndarray] = [],
                       parameters_list: List[float] = []) -> List[np.ndarray]:
        """
        プロパゲータを計算して返す
        
        Args:
            pulses_list: パルスのリスト
            time_grids_list: 時間グリッドのリスト
            parameters_list: パラメータのリスト
            
        Returns:
            プロパゲータのリスト
        """
        # 時間ステップを設定
        n_steps = len(pulses_list[0])
        dt = self.total_time / n_steps
        
        # 各時間ステップでのプロパゲータを計算
        for t in range(n_steps):
            # 各時間ステップでのハミルトニアン
            h_total = self.H_drift.copy()
            for i, h_control in enumerate(self.H_controls):
                if i < len(pulses_list):
                    h_total = h_total + pulses_list[i][t] * h_control
            
            # プロパゲータの計算 (U = exp(-i*H*dt))
            # SciPyのexpm関数を使用
            self.prop_store[t] = expm(-1j * h_total * dt)
        
        self.propagators_are_computed = True
        return self.prop_store
    
    def calculate_fidelity(self, final_state: np.ndarray) -> float:
        """
        最終状態と目標状態の間の忠実度を計算
        
        Args:
            final_state: 最終状態（密度行列）
            
        Returns:
            忠実度の値（0〜1）
        """
        # 密度行列間の忠実度 Tr(sqrt(sqrt(ρ1)ρ2sqrt(ρ1)))
        from scipy.linalg import sqrtm
        
        # 目標状態と最終状態が密度行列であることを確認
        target = self.target_state
        
        try:
            # 行列のトレースを使った忠実度計算
            sqrt_target = sqrtm(target)
            fidelity = np.abs(np.trace(sqrtm(sqrt_target @ final_state @ sqrt_target))) ** 2
            return float(np.real(fidelity))
        except:
            # フォールバック：単純なオーバーラップ
            return float(np.real(np.abs(np.trace(target @ final_state)) ** 2))
    
    def get_FoM(self, 
                pulses: List[np.ndarray] = [], 
                parameters: List[float] = [], 
                timegrids: List[np.ndarray] = []) -> Dict[str, float]:
        """
        評価関数（Figure of Merit）を計算
        
        Args:
            pulses: パルスのリスト
            parameters: パラメータのリスト
            timegrids: 時間グリッドのリスト
            
        Returns:
            評価関数の値を含む辞書
        """
        # プロパゲータを計算
        if not self.propagators_are_computed:
            self.get_propagator(pulses_list=pulses, time_grids_list=timegrids, parameters_list=parameters)
        
        self.propagators_are_computed = False
        
        # 最終プロパゲータを計算（時間順序積）
        u_final = np.eye(2, dtype=np.complex128)
        for u in self.prop_store:
            u_final = u @ u_final
        
        # 初期状態を最終状態に発展させる（密度行列の場合）
        final_state = u_final @ self.initial_state @ u_final.conj().T
        
        # 忠実度を計算
        fidelity = self.calculate_fidelity(final_state)
        
        # 最適化の方向に応じて評価関数の値を調整
        # 最大化問題の場合は負の値を返す（scipy.optimize.minimizeは最小化する）
        fom = self.FoM_factor * fidelity
        
        return {"FoM": fom}