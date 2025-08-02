module FrontendIR (FrontendIR (..)) where

import           Tensor (Tensor)

-- TODO: 本当は、このGADsの出力はTensorであるべき。また、TensorにIRの情報を持たせるべき
data FrontendIR where
    Add :: Tensor -> Tensor -> FrontendIR
